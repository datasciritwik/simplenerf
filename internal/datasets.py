import abc
import json
import os
from internal import camera_utils
from internal import configs
from internal import image as lib_image
from internal import raw_utils
from internal import utils
import numpy as np
import torch
from tqdm import tqdm


def load_dataset(split, train_dir, config: configs.Config):
    """Loads a split of a dataset using the data_loader specified by `config`."""
    dataset_dict = {
        'blender': Blender
    }
    return dataset_dict[config.dataset_loader](split, train_dir, config)


class Dataset(torch.utils.data.Dataset):
    """Dataset Base Class.

  Base class for a NeRF dataset. Creates batches of ray and color data used for
  training or rendering a NeRF model.

  Each subclass is responsible for loading images and camera poses from disk by
  implementing the _load_renderings() method. This data is used to generate
  train and test batches of ray + color data for feeding through the NeRF model.
  The ray parameters are calculated in _generate_rays().

  The public interface mimics the behavior of a standard machine learning
  pipeline dataset provider that can provide infinite batches of data to the
  training/testing pipelines without exposing any details of how the batches are
  loaded/created or how this is parallelized. Therefore, the initializer runs
  all setup, including data loading from disk using _load_renderings(), and
  begins the thread using its parent start() method. After the initializer
  returns, the caller can request batches of data straight away.

  The internal self._queue is initialized as queue.Queue(3), so the infinite
  loop in run() will block on the call self._queue.put(self._next_fn()) once
  there are 3 elements. The main thread training job runs in a loop that pops 1
  element at a time off the front of the queue. The Dataset thread's run() loop
  will populate the queue with 3 elements, then wait until a batch has been
  removed and push one more onto the end.

  This repeats indefinitely until the main thread's training loop completes
  (typically hundreds of thousands of iterations), then the main thread will
  exit and the Dataset thread will automatically be killed since it is a daemon.

  Attributes:
    alphas: np.ndarray, optional array of alpha channel data.
    cameras: tuple summarizing all camera extrinsic/intrinsic/distortion params.
    camtoworlds: np.ndarray, a list of extrinsic camera pose matrices.
    camtype: camera_utils.ProjectionType, fisheye or perspective camera.
    data_dir: str, location of the dataset on disk.
    disp_images: np.ndarray, optional array of disparity (inverse depth) data.
    distortion_params: dict, the camera distortion model parameters.
    exposures: optional per-image exposure value (shutter * ISO / 1000).
    far: float, far plane value for rays.
    focal: float, focal length from camera intrinsics.
    height: int, height of images.
    images: np.ndarray, array of RGB image data.
    metadata: dict, optional metadata for raw datasets.
    near: float, near plane value for rays.
    normal_images: np.ndarray, optional array of surface normal vector data.
    pixtocams: np.ndarray, one or a list of inverse intrinsic camera matrices.
    pixtocam_ndc: np.ndarray, the inverse intrinsic matrix used for NDC space.
    poses: np.ndarray, optional array of auxiliary camera pose data.
    rays: utils.Rays, ray data for every pixel in the dataset.
    render_exposures: optional list of exposure values for the render path.
    render_path: bool, indicates if a smooth camera path should be generated.
    size: int, number of images in the dataset.
    split: str, indicates if this is a "train" or "test" dataset.
    width: int, width of images.
  """

    def __init__(self,
                 split: str,
                 data_dir: str,
                 config: configs.Config):
        super().__init__()

        # Initialize attributes
        self._patch_size = max(config.patch_size, 1)
        self._batch_size = config.batch_size // config.world_size
        if self._patch_size ** 2 > self._batch_size:
            raise ValueError(f'Patch size {self._patch_size}^2 too large for ' +
                             f'per-process batch size {self._batch_size}')
        self._batching = utils.BatchingMethod(config.batching)
        self._use_tiffs = config.use_tiffs
        self._load_disps = config.compute_disp_metrics
        self._load_normals = config.compute_normal_metrics
        self._num_border_pixels_to_mask = config.num_border_pixels_to_mask
        self._apply_bayer_mask = config.apply_bayer_mask
        self._render_spherical = False

        self.config = config
        self.global_rank = config.global_rank
        self.world_size = config.world_size
        self.split = utils.DataSplit(split)
        self.data_dir = data_dir
        self.near = config.near
        self.far = config.far
        self.render_path = config.render_path
        self.distortion_params = None
        self.disp_images = None
        self.normal_images = None
        self.alphas = None
        self.poses = None
        self.pixtocam_ndc = None
        self.metadata = None
        self.camtype = camera_utils.ProjectionType.PERSPECTIVE
        self.exposures = None
        self.render_exposures = None

        # Providing type comments for these attributes, they must be correctly
        # initialized by _load_renderings() (see docstring) in any subclass.
        self.images: np.ndarray = None
        self.camtoworlds: np.ndarray = None
        self.pixtocams: np.ndarray = None
        self.height: int = None
        self.width: int = None

        # Load data from disk using provided config parameters.
        self._load_renderings(config)

        if self.render_path:
            if config.render_path_file is not None:
                with utils.open_file(config.render_path_file, 'rb') as fp:
                    render_poses = np.load(fp)
                self.camtoworlds = render_poses
            if config.render_resolution is not None:
                self.width, self.height = config.render_resolution
            if config.render_focal is not None:
                self.focal = config.render_focal
            if config.render_camtype is not None:
                if config.render_camtype == 'pano':
                    self._render_spherical = True
                else:
                    self.camtype = camera_utils.ProjectionType(config.render_camtype)

            self.distortion_params = None
            self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                       self.height)

        self._n_examples = self.camtoworlds.shape[0]

        self.cameras = (self.pixtocams,
                        self.camtoworlds,
                        self.distortion_params,
                        self.pixtocam_ndc)

        # Seed the queue with one batch to avoid race condition.
        if self.split == utils.DataSplit.TRAIN and not config.compute_visibility:
            self._next_fn = self._next_train
        else:
            self._next_fn = self._next_test

    @property
    def size(self):
        return self._n_examples

    def __len__(self):
        if self.split == utils.DataSplit.TRAIN and not self.config.compute_visibility:
            return 1000
        else:
            return self._n_examples

    @abc.abstractmethod
    def _load_renderings(self, config):
        """Load images and poses from disk.

    Args:
      config: utils.Config, user-specified config parameters.
    In inherited classes, this method must set the following public attributes:
      images: [N, height, width, 3] array for RGB images.
      disp_images: [N, height, width] array for depth data (optional).
      normal_images: [N, height, width, 3] array for normals (optional).
      camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
      poses: [..., 3, 4] array of auxiliary pose data (optional).
      pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
      distortion_params: dict, camera lens distortion model parameters.
      height: int, height of images.
      width: int, width of images.
      focal: float, focal length to use for ideal pinhole rendering.
    """

    def _make_ray_batch(self,
                        pix_x_int,
                        pix_y_int,
                        cam_idx,
                        lossmult=None
                        ):
        """Creates ray data batch from pixel coordinates and camera indices.

    All arguments must have broadcastable shapes. If the arguments together
    broadcast to a shape [a, b, c, ..., z] then the returned utils.Rays object
    will have array attributes with shape [a, b, c, ..., z, N], where N=3 for
    3D vectors and N=1 for per-ray scalar attributes.

    Args:
      pix_x_int: int array, x coordinates of image pixels.
      pix_y_int: int array, y coordinates of image pixels.
      cam_idx: int or int array, camera indices.
      lossmult: float array, weight to apply to each ray when computing loss fn.

    Returns:
      A dict mapping from strings utils.Rays or arrays of image data.
      This is the batch provided for one NeRF train or test iteration.
    """

        broadcast_scalar = lambda x: np.broadcast_to(x, pix_x_int.shape)[..., None]
        ray_kwargs = {
            'lossmult': broadcast_scalar(1.) if lossmult is None else lossmult,
            'near': broadcast_scalar(self.near),
            'far': broadcast_scalar(self.far),
            'cam_idx': broadcast_scalar(cam_idx),
        }
        # Collect per-camera information needed for each ray.
        if self.metadata is not None:
            # Exposure index and relative shutter speed, needed for RawNeRF.
            for key in ['exposure_idx', 'exposure_values']:
                idx = 0 if self.render_path else cam_idx
                ray_kwargs[key] = broadcast_scalar(self.metadata[key][idx])
        if self.exposures is not None:
            idx = 0 if self.render_path else cam_idx
            ray_kwargs['exposure_values'] = broadcast_scalar(self.exposures[idx])
        if self.render_path and self.render_exposures is not None:
            ray_kwargs['exposure_values'] = broadcast_scalar(
                self.render_exposures[cam_idx])

        pixels = dict(pix_x_int=pix_x_int, pix_y_int=pix_y_int, **ray_kwargs)

        # Slow path, do ray computation using numpy (on CPU).
        batch = camera_utils.cast_ray_batch(self.cameras, pixels, self.camtype)
        batch['cam_dirs'] = -self.camtoworlds[ray_kwargs['cam_idx'][..., 0]][..., :3, 2]

        # import trimesh
        # pts = batch['origins'][..., None, :] + batch['directions'][..., None, :] * np.linspace(0, 1, 5)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test.ply", "ply")
        #
        # pts = batch['origins'][0, 0, None, :] - self.camtoworlds[cam_idx][:, 2] * np.linspace(0, 1, 100)[:, None]
        # trimesh.Trimesh(vertices=pts.reshape(-1, 3)).export("test2.ply", "ply")

        if not self.render_path:
            batch['rgb'] = self.images[cam_idx, pix_y_int, pix_x_int]
        if self._load_disps:
            batch['disps'] = self.disp_images[cam_idx, pix_y_int, pix_x_int]
        if self._load_normals:
            batch['normals'] = self.normal_images[cam_idx, pix_y_int, pix_x_int]
            batch['alphas'] = self.alphas[cam_idx, pix_y_int, pix_x_int]
        return {k: torch.from_numpy(v.copy()).float() if v is not None else None for k, v in batch.items()}

    def _next_train(self, item):
        """Sample next training batch (random rays)."""
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        num_patches = self._batch_size // self._patch_size ** 2
        lower_border = self._num_border_pixels_to_mask
        upper_border = self._num_border_pixels_to_mask + self._patch_size - 1
        # Random pixel patch x-coordinates.
        pix_x_int = np.random.randint(lower_border, self.width - upper_border,
                                      (num_patches, 1, 1))
        # Random pixel patch y-coordinates.
        pix_y_int = np.random.randint(lower_border, self.height - upper_border,
                                      (num_patches, 1, 1))
        # Add patch coordinate offsets.
        # Shape will broadcast to (num_patches, _patch_size, _patch_size).
        patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(
            self._patch_size, self._patch_size)
        pix_x_int = pix_x_int + patch_dx_int
        pix_y_int = pix_y_int + patch_dy_int
        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
        else:
            cam_idx = np.random.randint(0, self._n_examples, (1,))

        if self._apply_bayer_mask:
            # Compute the Bayer mosaic mask for each pixel in the batch.
            lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
        else:
            lossmult = None

        return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx,
                                    lossmult=lossmult)

    def generate_ray_batch(self, cam_idx: int):
        """Generate ray batch for a specified camera in the dataset."""
        if self._render_spherical:
            camtoworld = self.camtoworlds[cam_idx]
            rays = camera_utils.cast_spherical_rays(
                camtoworld, self.height, self.width, self.near, self.far)
            return rays
        else:
            # Generate rays for all pixels in the image.
            pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
                self.width, self.height)
            return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

    def _next_test(self, item):
        """Sample next test batch (one full image)."""
        return self.generate_ray_batch(item)

    def collate_fn(self, item):
        return self._next_fn(item[0])

    def __getitem__(self, item):
        return self._next_fn(item)



class Blender(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, config):
        """Load images from disk."""
        if config.render_path:
            raise ValueError('render_path cannot be used for the blender dataset.')
        pose_file = os.path.join(self.data_dir, f'transform.json')
        with utils.open_file(pose_file, 'r') as fp:
            meta = json.load(fp)
        images = []
        disp_images = []
        normal_images = []
        cams = []
        for idx, frame in enumerate(tqdm(meta['frames'], desc='Loading Blender dataset', disable=self.global_rank != 0, leave=False)):
            fprefix = os.path.join(self.data_dir, frame['file_path'])

            def get_img(f, fprefix=fprefix):
                image = utils.load_img(fprefix + f)
                if config.factor > 1:
                    image = lib_image.downsample(image, config.factor)
                return image

            if self._use_tiffs:
                channels = [get_img(f'_{ch}.tiff') for ch in ['R', 'G', 'B', 'A']]
                # Convert image to sRGB color space.
                image = lib_image.linear_to_srgb_np(np.stack(channels, axis=-1))
            else:
                image = get_img('.png') / 255.
            images.append(image)

            if self._load_disps:
                disp_image = get_img('_disp.tiff')
                disp_images.append(disp_image)
            if self._load_normals:
                normal_image = get_img('_normal.png')[..., :3] * 2. / 255. - 1.
                normal_images.append(normal_image)

            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))

        self.images = np.stack(images, axis=0)
        if self._load_disps:
            self.disp_images = np.stack(disp_images, axis=0)
        if self._load_normals:
            self.normal_images = np.stack(normal_images, axis=0)
            self.alphas = self.images[..., -1]

        rgb, alpha = self.images[..., :3], self.images[..., -1:]    ### comment this 
        self.images = rgb * alpha + (1. - alpha)  # Use a white background. and this line for original background images.
        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = np.stack(cams, axis=0)
        self.focal = .5 * self.width / np.tan(.5 * float(meta['camera_angle_x']))
        self.pixtocams = camera_utils.get_pixtocam(self.focal, self.width,
                                                   self.height)

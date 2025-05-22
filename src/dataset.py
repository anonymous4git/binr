import torch 
from e3nn.o3 import xyz_to_angles, angles_to_xyz 
from torch.utils.data import Dataset 


class DWIReprDataset(Dataset):

    def __init__(self, 
                 image, 
                 mask, 
                 bvecs, 
                 bvals, 
                 normalize_in_out=True, 
                 normalize_0_1=False, 
                 subsampl_step=1, 
                 bval_selection='all',
                 max_bval=3000,
                 cfg_precache_hash=None,
                 preload_to_gpu=False,
                 device='cpu',
                 is_train=True, 
                 dwi_headers=None):

        super().__init__()
        self.device = device
        self.normalize_in_out = normalize_in_out
        self.normalize_0_1 = normalize_0_1
        self.storage_precicion = torch.float16
        self.element_precicion = torch.float32
        self.eps = 1e-8
        self.y_min, self.y_max = image.min(), image.max()
        self.num_slices = image.shape[2]
        self.cfg_precache_hash = cfg_precache_hash
        self.max_bval = max_bval
        self.dwi_headers = dwi_headers
        image = self._subsample_spatial(image, subsampl_step)
        mask = self._subsample_spatial(mask, subsampl_step)

        bvecs, bvals, image = self._select_b(bvecs, bvals, image, bval_selection)        

        self.image = image.to(self.device, dtype=self.storage_precicion)
        self.mask = mask.to(self.device, dtype=torch.bool)

        self.bvecs = bvecs.to(self.device, dtype=self.storage_precicion)
        self.bvals = bvals.to(self.device, dtype=self.storage_precicion)
        # check if bvals are non-negative
        if (self.bvals<0).any():    
            print(f"bvals: {self.bvals}")
            raise ValueError("b-values are negative")
        self.bparams = torch.cat([self.bvals[:, None], self.bvecs], dim=-1)
        self.dim = self.image.shape
        self.num_samples = torch.prod(torch.tensor(self.dim)).item()
        
        print(f"dim: {self.dim}")
        self.xyz_grid = None 
        self.x, self.y = None, None
        # self.normalize_in_out = normalize_in_out
        self.y_min, self.y_max = self.image.min(), self.image.max()

        self._init_repr(self.image, self.mask, self.bvecs, self.bvals, is_train=is_train, 
                        precicion=self.element_precicion)

        # pre-compute normalization 
        if self.normalize_in_out:
            # self.x_norm, self.y_norm = None, None
            if self.normalize_0_1:
                self.x_norm, self.y_norm = self.normalize_xy_01(self.x, self.y)
            else:
                raise NotImplementedError
        else:
            self.x_norm, self.y_norm = self.x, self.y

        # shuffle once
        perm = torch.randperm(self.x_norm.shape[0])
        self.x_norm = self.x_norm[perm]
        self.y_norm = self.y_norm[perm]

        if preload_to_gpu:
            self.x_norm = self.x_norm.to('cuda')
            self.y_norm = self.y_norm.to('cuda')


    def _init_repr(self, image, mask, bvecs, bvals, is_train=True, precicion=None):

        if precicion == None:
            precicion = self.storage_precicion

        image = image.to(self.device, dtype=precicion)
        mask = mask.to(self.device, dtype=torch.bool)
        bvecs = bvecs.to(self.device, dtype=precicion)
        bvals = bvals.to(self.device, dtype=precicion)

        bparams = torch.cat([bvals[:, None], bvecs], dim=-1)
        dim = image.shape
        xyz_grid = None

        x_unfold, y_unfold, xyz_grid = self._unfold_xyz(image, bparams)
        mask_unfold = self._unfold_mask(mask, image)

        # Mask values
        x_masked, y_masked = x_unfold[mask_unfold], y_unfold[mask_unfold]
        print(f"Loaded {x_masked.shape[0]} {'training' if is_train else 'validation'} points")

        # Set self.x and self.y regardless of train/val mode
        self.x, self.y = x_masked, y_masked
        self.dim, self.xyz_grid = dim, xyz_grid
        
        if is_train:
            self.image = image
            self.bvecs, self.bvals, self.bparams = bvecs, bvals, bparams
            self.dim, self.xyz_grid =  dim, xyz_grid
        else:
            self.image_val = image
            self.bvecs_val, self.bvals_val, self.bparams_val = bvecs, bvals, bparams
            self.dim_val, self.xyz_grid_val = dim, xyz_grid
            self.x_val, self.y_val = x_masked, y_masked


    def _unfold_xyz(self, image, bparams):
        dim = image.shape
        n_coord = dim[0] * dim[1] * dim[2]
        n_c = dim[-1]

        # reshape into [dimx x dimy x dimz, dimc]
        xyz_grid = torch.stack(torch.meshgrid(
                torch.arange(dim[0]),
                torch.arange(dim[1]),
                torch.arange(dim[2]), indexing='ij'), 
                dim=-1).to(self.device, dtype=self.element_precicion)

        xyz_coord = xyz_grid.reshape(-1, 3)

        # broadcast bval/bvec channel [n_coord, C]
        bparams = bparams[None,].expand(n_coord, -1, -1).reshape(-1, 4)  

        # repeat coords [n_coord x n_c, 3]
        xyz_coords_expanded = xyz_coord.repeat_interleave(n_c, dim=0)

        # collect indices 
        ret_x = torch.column_stack([xyz_coords_expanded, bparams])

        # reform also the y
        # ret_y = image.view(-1)
        ret_y = image.reshape(-1)
        # add assert that b-values are not negative
        assert torch.all(ret_x[:, 3] >= 0), 'b-values are negative'

        return ret_x, ret_y, xyz_grid

    @staticmethod
    def _subsample_spatial(img, subsampl_step):
        return img[::subsampl_step, ::subsampl_step, ::subsampl_step]

    @staticmethod
    def _select_b(bvecs, bvals, image, bval_selection, bval_tol=50):
        if bval_selection == 'all':
            return bvecs, bvals, image
        
        assert isinstance(bval_selection, list), "please provide a list of bvals to include or set to all"

        bval_sel = torch.tensor(bval_selection)
        diff = torch.abs(bvals.unsqueeze(1) - bval_sel)  
        mask = (diff <= bval_tol).any(dim=1)
        bval_idx = torch.nonzero(mask, as_tuple=True)[0]

        return bvecs[bval_idx], bvals[bval_idx], image[..., bval_idx]

    def normalize_xy_01(self, x, y, inverse=False):
        # Upcast to element precision to prevent rounding errors
        x = x.to(self.element_precicion)
        y = y.to(self.element_precicion)

        # normalize x between -1 and 1
            # Normalize x
        x_n, y_n = torch.empty_like(x), torch.empty_like(y)
        if not inverse:
            x_n[..., :3] = 2 * (x[..., :3] / torch.tensor(self.dim[:3], device=x.device)) - 1
            x_n[..., 3] = x[..., 3] / self.max_bval
            x_n[..., 4:] = x[..., 4:]

        else:
            x_n[..., :3] = ((x[..., :3] + 1) / 2) * torch.tensor(self.dim[:3], device=x.device)
            x_n[..., 3] = x[..., 3] * self.max_bval
            x_n[..., 4:] = x[..., 4:]

        y_n = (y - self.y_min) / (self.y_max - self.y_min) if not inverse else (y * (self.y_max - self.y_min)) + self.y_min

        # Downcast to storage precision for return
        return x_n, y_n
        # return x_n.to(self.storage_precicion), y_n.to(self.storage_precicion)


    @staticmethod
    def _unfold_bvals_bvecs(image, bvecs, bvals):
        pass

    def _fold_xyz(self, x, y, skip_checks=False, denormalize=False, force_int_coords=False):

        if denormalize:
            if self.normalize_0_1:
                x, y = self.normalize_xy_01(x, y, inverse=True)
            else:
                raise NotImplementedError

        if force_int_coords:
            x[:, :3] = x[:, :3].round()

        if not skip_checks:
            assert self.xyz_grid is not None

            # test if all x,y,z values are integers
            assert torch.all(x[:, :3] == x[:, :3].int())

            # test if all x,y,z values are within the image grid
            assert self._vectors_in(x[:, :3], self.xyz_grid.reshape(-1, 3))

            # test if all bparams are within the bvecs, bvals
            assert self._vectors_in(x[:, 3:], self.bparams)

        # remove extra dim 
        y = y.squeeze()
        y = y.to(x.device)

        # Create image_unfolded on the same device and dtype as x
        image_unfolded = torch.zeros_like(self.image, device=x.device, dtype=x.dtype)

        # Convert y to the same dtype as image_unfolded
        y = y.to(image_unfolded.dtype)

        # recover the indices of the bparams 
        bparam_idx = self._map_to_bparam_idx(x[:, 3:])

        image_unfolded[x[:, 0].round().int(), 
                       x[:, 1].round().int(), 
                       x[:, 2].round().int(),
                       bparam_idx.int()] = y

        return image_unfolded


    def _unfold_mask(self, mask, image):

        mask_expanded = mask[..., None].repeat(1, 1, 1, image.shape[3])
        # ret_mask = mask_expanded.view(-1)
        ret_mask = mask_expanded.reshape(-1)

        return ret_mask


    @staticmethod
    def _vectors_in(A, B, atol=1e-6):
        # Ensure both tensors are on same device
        if A.device != B.device:
            B = B.to(A.device)

        diff = A.unsqueeze(1) - B.unsqueeze(0)  # [N, M, 4]
        matches = torch.all(torch.abs(diff) <= atol, dim=2)  # [N, M]

        return torch.all(matches.any(dim=1))

    
    def _map_to_bparam_idx(self, A, atol=1e-6):
        # Make sure both tensors are on the same device
        if A.device != self.bparams.device:
            # Either move A to bparams device or vice versa
            # Since bparams is typically smaller, we'll move it to A's device
            bparams_device = self.bparams.to(A.device)
        else:
            bparams_device = self.bparams
        
        diff = A.unsqueeze(1) - bparams_device.unsqueeze(0)  # [N, M, 4]
        matches = torch.all(torch.abs(diff) <= atol, dim=2)  # [N, M]
        indices = torch.full((A.shape[0],), -1, dtype=torch.long, device=A.device)  
        for i in range(A.shape[0]):
            match_idx = torch.where(matches[i])[0]  
            if len(match_idx) > 0:
                indices[i] = match_idx[0]  

        return indices


    def __len__(self):
        assert self.x is not None, 'call DWIReprDataset._init_repr first'

        return self.x.shape[0]       


    def __getitem__(self, idx, train=True):
        assert self.x is not None, 'call DWIReprDataset._init_repr first'

        # Select train or validation data
        if train:
            x = self.x_norm[idx]
            y = self.y_norm[idx].unsqueeze(-1)

            return x, y, idx

        else:
            x = self.x_val[idx]
            y = self.y_val[idx].unsqueeze(-1)

            # Normalize
            if self.normalize_in_out:
                if self.normalize_0_1:
                    x_norm, y_norm = self.normalize_xy_01(x, y)
                else:
                    x_norm, y_norm = self.normalize_xy_meanstd(x, y)

                return x_norm, y_norm, idx

                # assert torch.all(x[:, 3] >= 0), 'b-values are negative'
            else:
                return x, y, idx



class PreBatchedDWIReprDataset(DWIReprDataset):
    """
    hacky method for fast indexing on gpu
    """

    def __init__(self, *args, train_batch_size=8192, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.x_norm is not None and self.y_norm is not None, "x_norm/y_norm must be precomputed"

        N = self.x_norm.shape[0]
        B = train_batch_size
        remainder = N % B

        if remainder != 0:
            pad_len = B - remainder
            self.x_norm = torch.cat([self.x_norm, self.x_norm[:pad_len]], dim=0)
            self.y_norm = torch.cat([self.y_norm, self.y_norm[:pad_len]], dim=0)        

        self.num_batches = self.x_norm.shape[0] // B
        self.x_batches = self.x_norm.view(self.num_batches, B, -1)
        self.y_batches = self.y_norm.view(self.num_batches, B, -1)

        # self.num_batches = N // B

        # self.x_batches = self.x_norm[:self.num_batches * B].view(self.num_batches, B, -1)
        # self.y_batches = self.y_norm[:self.num_batches * B].view(self.num_batches, B, -1)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        x, y = self.x_batches[idx], self.y_batches[idx]

        return x, y, idx

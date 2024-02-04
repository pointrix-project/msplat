
import torch
import DifferentiablePointRender.GaussianSplatting as gs

if __name__ == "__main__":
    
    print("=============================== running test on sort_gaussian ===============================")

    w = 32
    h = 16
    uv = torch.zeros((4, 2), dtype=torch.float32, device="cuda")
    depth = torch.zeros((4, 1), dtype=torch.float32, device="cuda")
    radii = torch.zeros((4, 1), dtype=torch.int, device="cuda")
    num_tiles_hit = torch.zeros((4, 1), dtype=torch.int, device="cuda")

    uv[0, 0] = 2
    uv[0, 1] = 2
    depth[0, 0] = 1.0
    radii[0, 0] = 2
    num_tiles_hit[0, 0] = 1

    uv[1, 0] = 30
    uv[1, 1] = 2
    depth[1, 0] = 2.0
    radii[1, 0] = 8
    num_tiles_hit[1, 0] = 1

    uv[2, 0] = 8
    uv[2, 1] = 8
    depth[2, 0] = 2.0
    radii[2, 0] = 16
    num_tiles_hit[2, 0] = 2

    uv[3, 0] = 30
    uv[3, 1] = 2
    depth[3, 0] = 3.0
    radii[3, 0] = 1
    num_tiles_hit[3, 0] = 1

    gaussian_ids_sorted, tile_bins = gs.sort_gaussian(uv, depth, w, h, radii, num_tiles_hit)

    target_gaussian_ids_sorted = torch.tensor([0, 2, 2, 1, 3], device='cuda:0', dtype=torch.int32)
    target_tile_bins = torch.tensor([[0, 2], [2, 5]], device='cuda:0', dtype=torch.int32)
    
    torch.testing.assert_close(gaussian_ids_sorted, target_gaussian_ids_sorted)
    torch.testing.assert_close(tile_bins, target_tile_bins)
    print(" Pass.")

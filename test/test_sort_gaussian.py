
import torch
import msplat as ms

if __name__ == "__main__":
    
    print("=============================== running test on sort_gaussian ===============================")

    w = 32
    h = 16
    uv = torch.zeros((4, 2), dtype=torch.float32, device="cuda")
    depth = torch.zeros((4, 1), dtype=torch.float32, device="cuda")
    radius = torch.zeros((4, 1), dtype=torch.int, device="cuda")
    tiles = torch.zeros((4, 1), dtype=torch.int, device="cuda")

    uv[0, 0] = 2
    uv[0, 1] = 2
    depth[0, 0] = 1.0
    radius[0, 0] = 2
    tiles[0, 0] = 1

    uv[1, 0] = 30
    uv[1, 1] = 2
    depth[1, 0] = 2.0
    radius[1, 0] = 8
    tiles[1, 0] = 1

    uv[2, 0] = 8
    uv[2, 1] = 8
    depth[2, 0] = 1.5
    radius[2, 0] = 16
    tiles[2, 0] = 2

    uv[3, 0] = 30
    uv[3, 1] = 2
    depth[3, 0] = 3.0
    radius[3, 0] = 1
    tiles[3, 0] = 1

    idx_sorted, tile_range = ms.sort_gaussian2(uv, depth, w, h, radius, tiles)

    target_idx_sorted = torch.tensor([0, 2, 2, 1, 3], device='cuda:0', dtype=torch.int32)
    target_tile_range = torch.tensor([[0, 2], [2, 5]], device='cuda:0', dtype=torch.int32)
    
    print(idx_sorted)
    print(target_idx_sorted)
    
    print(tile_range)
    print(target_tile_range)
    
    torch.testing.assert_close(idx_sorted, target_idx_sorted)
    torch.testing.assert_close(tile_range, target_tile_range)
    print(" Pass.")

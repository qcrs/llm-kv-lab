"""Project-owned read-only oracle for pinned MRV2 BlockTables behavior."""

import os

import torch

from vllm.v1.worker.gpu.block_table import BlockTables


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    device = torch.device("cuda")
    bt = BlockTables(
        block_sizes=[16],
        max_num_reqs=1,
        max_num_batched_tokens=32,
        max_num_blocks_per_group=[8],
        device=device,
        kernel_block_sizes=[16],
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
    )

    bt.append_block_ids(0, ([10, 11, 12, 13, 14, 15, 16, 17],), overwrite=True)
    bt.apply_staged_writes()
    print("initial_num_blocks", bt.num_blocks.np.tolist())
    print("initial_row", bt.block_tables[0].gpu[0].cpu().tolist())

    bt.append_block_ids(0, ([10, 11, 16, 17],), overwrite=True)
    bt.apply_staged_writes()
    print("short_num_blocks", bt.num_blocks.np.tolist())
    print("short_row", bt.block_tables[0].gpu[0].cpu().tolist())

    gathered = bt.gather_block_tables(torch.tensor([0], device=device), 1)
    print("gathered_row", gathered[0][0].cpu().tolist())

    positions = torch.tensor([0, 1, 16, 17, 32, 33, 48, 49], device=device)
    query_start_loc = torch.tensor([0, positions.numel()], device=device)
    slots = bt.compute_slot_mappings(
        torch.tensor([0], device=device),
        query_start_loc,
        positions,
        positions.numel(),
    )
    print("positions", positions.cpu().tolist())
    print("slot_mappings", slots[0].cpu().tolist())

    effective_len = 70
    print("partial_tail", {
        "effective_kv_len": effective_len,
        "block_size": 16,
        "physical_block_count": (effective_len + 15) // 16,
        "last_block_offset": effective_len % 16,
    })


if __name__ == "__main__":
    main()

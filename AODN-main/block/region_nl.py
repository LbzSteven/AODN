import torch
from torch import nn
from block.NLblock import NONLocalBlock2D


class RegionNONLocalBlock(nn.Module):
    def __init__(self, in_channels, grid=[4, 4]):
        super(RegionNONLocalBlock, self).__init__()

        self.non_local_block = NONLocalBlock2D(in_channels, sub_sample=True, bn_layer=False)
        self.grid = grid

    def forward(self, x):
        batch_size, _, height, width = x.size()
        # 拆分成gird第一个量（4）个row
        input_row_list = x.chunk(self.grid[0], dim=2)

        output_row_list = []
        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)

            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):

                grid = self.non_local_block(grid)
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)
        return output


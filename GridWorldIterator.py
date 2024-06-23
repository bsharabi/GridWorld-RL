import ast

class GridWorldIterator:
    def __init__(self, filename):
        self.filename = filename
        self.grids = self._parse_file()
        self.current_index = -1

    def _parse_file(self):
        grids = []
        with open(self.filename, 'r') as file:
            content = file.read()
            grid_blocks = content.split('\n\n')  # Split the file content into blocks by empty lines
            for block in grid_blocks:
                if block.strip():
                    grid = {}
                    for line in block.split('\n'):
                        line = line.strip()
                        if line.startswith('#') or not line:
                            continue
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            value = ast.literal_eval(value)
                        except ValueError:
                            pass
                        grid[key] = value
                    grids.append(grid)
        return grids

    def __iter__(self):
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index >= len(self.grids):
            raise StopIteration
        grid = self.grids[self.current_index]
        return grid['w'], grid['h'], grid['L'], grid['p'], grid['r']

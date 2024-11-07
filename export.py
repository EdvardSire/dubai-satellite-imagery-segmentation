import nbformat

def ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    with open(py_file, 'w', encoding='utf-8') as f:
        for cell in nb.cells:
            if cell.cell_type == 'code':
                f.write(cell.source + '\n\n')

ipynb_to_py('dubai-satellite-imagery-segmentation.ipynb', 'main.py')


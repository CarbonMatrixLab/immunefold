import sys
import shutil

infile = sys.argv[1]

with open(infile) as fin, open('./tmp.py', 'w') as fw:
    for line in fin:
        if line.startswith ('import carbon'):
            new_line = line.replace('import carbon', 'import carbonmatrix')
        elif line.startswith('from carbon'):
            new_line = line.replace('from carbon', 'from carbonmatrix')
        else:
            new_line = line

        fw.write(new_line)

shutil.copy('./tmp.py', infile)

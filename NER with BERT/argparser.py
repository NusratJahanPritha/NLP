import argparse

parser = argparse.ArgumentParser(description='Input file path for inference. Type has to be .txt or .csv or .tsv')
parser.add_argument('--input' ,help='provide input file path, kept in root directory')
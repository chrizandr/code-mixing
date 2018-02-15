"""."""
from data_handler import read_data, write_data, break_in_subword
import pdb

INPUT = "data/IITB.en-hi.hi.roman.clean"
OUTPUT = "data/IITB.en-hi.hi.syll"
print "Reading"
data = read_data(INPUT, encoding="UNI", clean=True)
print "Breaking"
new_data = break_in_subword(data, sentences=True)
print "Writing"
write_data(OUTPUT, new_data, encoding="UNI")
pdb.set_trace()


with open("corrupted_test.h5", "wb") as f:
    f.write(b"This is not a valid HDF5 file")

print("Corrupted test file created!")

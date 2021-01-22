# Archiving additional objects within the bundle object

All the object files (.o) stored in this directory will be automatically
serialized as C buffer arrays and stored within Glow library at build time.
When building bundles with Glow tools the user can opt to archive additional
objects within the bundle object file by specifying the following command line
option:
```
  -bundle-objects=<object1>,<object2>,...
```

For example if this directory contains the files *lib1.o* and *lib2.o* the
user can choose to archive them with the bundle object with the following command:
```
  -bundle-objects=lib1.o,lib2.o
```

The names of the object files stored in this directory must have the *.o* extension
and their names (without extension) must qualify to valid C identifiers.

The file *test.o* was obtained by compiling the following source code on x86_64 and
is used for testing:
```
const char *testMsg = "This is extra bundle object file for testing!";
```

from distutils.core import setup, Extension

module = Extension('BssGed',
	sources=['BssGedModule.cpp'],
	extra_objects=['bitmap.o', 'graph.o', 'verifyGraph.o','global.o' ,'treeNode.o', 'BSED.o'],
	include_dirs=['.'],
	language='c++',
	extra_compile_args=['-O3','-std=c++0x','-D_FILE_OFFSET_BITS=64','-D_LARGE_FILE'])

setup(name='BssGed',
	version='0.0',
	ext_modules=[module])

import setuptools

setuptools.setup(
    name='slowfast',
    version='0.0.1',
    description='',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Evangelos Kazakos, Iran Roman, Bea Steers',
    url=f'https://github.com/iranroman/ego_actrecog_analysis',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': []},
    install_requires=[
        'torch', 'numpy', 'fvcore',
        'torchvision',
        ## slowfast
        #'detectron2 @ git+ssh://git@github.com/facebookresearch/detectron2.git',
        # auditory slowfast
        'librosa',
        # omnivore
        'pandas', 

        # test
        'simplejson', 'matplotlib', 
        # ava dataset
        'opencv-python', 
        # kinetics
        'av',
    ],
    extras_require={
        # 'slowfast': ['detectron2'],
        # 'auditory_slowfast': ['librosa'],
        # 'mtcn': [],
    },
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License',
    keywords='slowfast auditory visual mtcn transformer multimodal temporal context action recognition kitchen verb noun epic-kitchens egocentric video')

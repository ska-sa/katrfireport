from setuptools import setup

setup(
    name='katrfireport',
    version='0.1.0',    
    description='This repository package an interactive bokeh-based RFI statistics report for MeerKAT observation',
    url='https://github.com/SihlanguI/katrfireport',
    author='Isaac Sihlangu',
    author_email='isihlangu@sarao.ac.za',
    license='',
    packages=['katrfireport'],
    install_requires=['bokeh==2.3.3',
                      'katdal==0.20.1',
                      'numpy==1.19.5',
                      'pandas==1.1.5'                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
    ],
)

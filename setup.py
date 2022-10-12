import setuptools

setuptools.setup(
   name="NPanalysis",
   version="1.3.dev1",
   author="Subhamoy Mahajan",
   description="Analysis module for two-component nanoparticles",
   url="https://github.com/subhamoymahajan/NPanalysis",
   license='GPLv3',
   install_requires=['numpy==1.19.2','networkx==2.5','matplotlib==3.3.2','numba==0.49.1'],
   packages=['NPanalysis'],
   package_data={'': ['LICENSE.txt']},
   classifiers=[
        "Development Status :: 5-Production/Stable",
        "Intended Audience :: Science/Research",
        "Indended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language:: Python :: 3.7",
        "Programming Language:: Unix Shell",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
   ],
   python_requires='>=3.7'
)


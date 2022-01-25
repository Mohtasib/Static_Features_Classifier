from setuptools import setup

setup(name='sfc',
      version='0.1.0',
      description="This is a static features classifier for Point-Could clusters using an Attention-RNN model",
      author="Abdalkarim Mohtasib",
      author_email='amohtasib@lincoln.ac.uk',
      platforms=["any"],
      license="GPLv3",
      url="https://github.com/Mohtasib/Static_Features_Classifier",
      install_requires=['numpy==1.19.2', 
			'pandas==1.1.5',
			'scikit-learn==0.24.2',
			'h5py==2.10.0',
			'tensorflow-gpu==1.14.0',
			'keras==2.2.5']  # And any other dependencies pepper env needs
)

# python==3.6.10

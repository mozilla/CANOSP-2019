from setuptools import find_packages, setup

setup(
    name="mozilla-fldp",
    use_scm_version=False,
    version="0.1.0",
    setup_requires=["setuptools_scm", "pytest-runner"],
    tests_require=["pytest"],
    include_package_data=True,
    packages=find_packages(exclude=["tests", "tests/*"]),
    description="Federated Learning Experimentation",
    author="Mozilla Corporation",
    author_email="dzeber@mozilla.com",
    url="https://github.com/mozilla/CANOSP-2019",
    license="MPL 2.0",
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment :: Mozilla",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    zip_safe=False,
)

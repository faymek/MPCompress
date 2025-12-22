# Poetry Environment Management

**Adding Packages**
```sh
poetry add <package>
```

# Git Workflow

The default branch is `dev`. Features should be developed until complete before being merged into the `main` branch.

Each contributor should create a new branch. For example:
```sh
git checkout -b dev/feature1
```

**How to submit:**
```sh
git push origin dev/feature2
```

After pushing, create a Pull Request on GitHub to merge your changes into the `dev` branch.

# Code Integration

For new projects, please place them under `examples/project/`. Once initial results are achieved, reusable components can be merged into the `mpcompress/` directory.

# Documentation

The documentation is built using [Zensical](https://zensical.org/) — the successor to [MkDocs Material](https://squidfunk.github.io/mkdocs-material/), currently in development.

To get started, install the required packages:

```sh
pip install zensical
pip install mkdocstrings-python
```

Then, launch the local documentation server with:

```sh
zensical serve
```

Once the server is running, you can view the documentation at [http://localhost:8000](http://localhost:8000).
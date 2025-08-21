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



We are currently developing our environments on an Ubuntu system. The operating system version is 20.04.3 LTS.

### Setting Up the Environment

1. **Create an environment (requires [Conda installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)):**

   Use the following command to create a new Conda environment named `omnirobust` with Python 3.10:

   ```bash
   conda create -n omnirobust python=3.10
   ```

   Activate the newly created environment:

   ```bash
   conda activate omnirobust
   ```

2. **Install dependency packages:**

   Install the necessary packages using pip. Make sure you are in the project directory where the `setup.py` file is located:

   ```bash
   pip install -e .
   ```

### Testing the Tasks

To run the tests, navigate to the `tests` directory and execute the test script:

```bash
cd examples/
python test.py
```

---

Ensure you follow these steps to set up and test the environment properly. Adjust paths and versions as necessary based on your specific setup requirements.

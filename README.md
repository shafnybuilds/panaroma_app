# Panaroma App

This repository contains the Panaroma App project, which uses the [uv](https://github.com/astral-sh/uv) package manager for fast Python dependency management and virtual environment handling.

## Prerequisites

Before you begin, ensure you have the following installed on your local machine:

- **Python** (version 3.8 or higher is recommended)
- [uv](https://github.com/astral-sh/uv) package manager  
  _Follow install instructions at: https://github.com/astral-sh/uv#installation_

## Cloning the Repository

1. Open your terminal or command prompt.
2. Clone the repository using the following command:

   ```bash
   git clone https://github.com/shafnybuilds/panaroma_app.git
   cd panaroma_app
   ```

## Setting Up Your Environment with uv

1. **Install dependencies**  
   This project uses uv to manage dependencies. After cloning, run:

   ```bash
   uv pip install -r requirements.txt
   ```

   - If your project uses a `pyproject.toml` or `requirements.in`, adjust the command as needed:

     ```bash
     uv pip install -r requirements.in
     # or
     uv pip install .
     # or
     uv pip install -e .
     ```

2. **Activate the virtual environment (optional)**  
   uv automatically manages isolated environments. If you want to explicitly activate the environment, you can use [uv venv](https://github.com/astral-sh/uv#venv):

   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

   > **Note:** You can skip this step if you do not need to activate a venv manually. uv will handle isolation itself.

## Running the Application

Run the application as per your entry point. For example, if your project uses a `main.py`:

```bash
uv pip run python main.py
```

Or, if you have defined an entry point in `pyproject.toml`, you can use:

```bash
uv pip run python -m <your_module>
```

_Replace `<your_module>` with the actual module name if needed._

## Adding or Updating Dependencies

To add a new package:

```bash
uv pip install <package-name>
```

To export the installed packages to `requirements.txt`:

```bash
uv pip freeze > requirements.txt
```

## Useful uv Commands

- **Install dependencies:**  
  `uv pip install -r requirements.txt`
- **Add a new dependency:**  
  `uv pip install <package-name>`
- **Update dependencies:**  
  `uv pip upgrade`
- **List installed packages:**  
  `uv pip list`
- **Create a virtual environment:**  
  `uv venv .venv`

## Troubleshooting

- Make sure you have the correct version of Python installed.
- If you encounter issues with uv, refer to the [uv GitHub documentation](https://github.com/astral-sh/uv).
- If packages fail to install, check your network connection and Python version compatibility.

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)

---

_This project uses [uv](https://github.com/astral-sh/uv) for Python dependency management and virtual environments for a faster and more reproducible workflow._

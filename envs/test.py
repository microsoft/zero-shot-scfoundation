#! /usr/bin/env python

try:
    from sc_foundation_evals.helpers.custom_logging import log
except ImportError:
    import logging

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)
    msg = "Cannot load sc_foundation_evals custom logging module. Exiting..."
    log.error(msg)
    raise ImportError(msg)

log.info("Hello from the test script! This is to test the build process.")


def import_package(package_name):
    """
    Try to import a package and return the package if successful.
    Logs and raises an error if the package is not available.
    """
    try:
        package = __import__(package_name)
        version = getattr(package, "__version__", None)
        log.info(
            f"Successfully imported {package_name}. "
            f"Version: {version if version else 'unknown'}"
        )
        return package

    except ImportError as e:
        msg = f"Could not import required package: {package_name}"
        log.error(f"{msg}: {e}")
        raise ImportError(msg)


def test_cuda_availability():
    """
    Check if CUDA is available and log the result.
    """
    torch = import_package("torch")
    if torch.cuda.is_available():
        log.info("Success -- CUDA is available!")
    else:
        log.error(
            "CUDA is not available. Please check your system configuration."
        )


def main():
    try:
        log.debug("Testing CUDA availability...")
        test_cuda_availability()
        log.debug("Testing loading scGPT...")
        import_package("scgpt")
        log.debug("Testing loading Geneformer...")
        import_package("geneformer")
        log.info("All tests passed successfully! :)")

    except Exception as e:
        log.error(f"An error occurred during the testing process: {e}")
        raise


if __name__ == "__main__":
    main()

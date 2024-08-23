FROM kzkedzierska/sc_foundation_evals:latest

# Install Jupyter Notebook
RUN pip install notebook

WORKDIR /workspace

# Expose the port Jupyter will run on
EXPOSE 8888

# Set the default command to run when starting the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
FROM julia:1.12.6-bookworm

# Prevent out-of-memory errors and optimize build on resource-constrained containers (like Render Free tier)
ENV JULIA_NUM_PRECOMPILE_TASKS=1
ENV JULIA_CPU_TARGET="generic"

WORKDIR /app

# Copy the package environment specification
COPY Project.toml Manifest.toml ./

# Install and precompile all packages in the Docker image
RUN julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Copy all source files
COPY . .

# Remove any temporary install script
RUN rm -f install_packages.jl

# Expose the server port
EXPOSE 8080

# Run the HTTP server
CMD ["julia", "--project", "server.jl"]

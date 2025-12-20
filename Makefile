all: ensure-build container-rm run

###############################
### Docker Builtin Commands ###
###############################

# Build the image
build:
	docker build -t barn-challenge-2026:latest .

# Save the image to a tar file
save:
	mkdir -p tmp/image
	docker save barn-challenge-2026:latest -o tmp/image/barn-challenge-2026.tar
	@echo "Image saved to tmp/image/barn-challenge-2026.tar"

# Load the image from a tar file
load:
	docker load -i tmp/image/barn-challenge-2026.tar

run:
	@if [ -z "$$(docker images -q barn-challenge-2026:latest)" ]; then \
		echo "Image barn-challenge-2026:latest not found. Please build the image first."; \
		exit 1; \
	fi
	xhost +local:root
	-docker run -it \
		--privileged \
		--env="DISPLAY" \
		-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
		-e XDG_RUNTIME_DIR=/tmp \
		-e QT_X11_NO_MITSHM=1 \
		--net=host \
		--name barn-challenge-2026 \
		--ulimit nofile=1024:524288 \
		--mount type=bind,source=$(CURDIR),target=/jackal_ws/src/the-barn-challenge \
		barn-challenge-2026:latest zsh
	xhost -local:root

# Start the container
start:
	-docker start barn-challenge-2026
	-docker exec -it barn-challenge-2026 /bin/zsh

# Stop the container
stop:
	-docker stop barn-challenge-2026

container-rm: stop
	-docker container rm barn-challenge-2026

rmi:
	-docker rmi barn-challenge-2026:latest

# Attach to the running container
attach:
	-docker exec -it barn-challenge-2026 /bin/zsh

#####################
### Tool Commands ###
#####################

# Remove the container and image
clean: stop container-rm rmi
	echo "Cleaned up the container and image."

### Build
# Build the image if network is present
#     If the cached image is different from the current image, replace the cached image
#     If the cached image is the same as the current image, do nothing
# If not, find the cached image
# If cached image is not present, check if the image is already present
# If all fail, exit with error code
ensure-build:
	@set -e; \
	if ping -c 1 -W 1 8.8.8.8 > /dev/null 2>&1; then \
		echo -e "\e[1;36mNetwork is up, building the image\e[0m"; \
		$(MAKE) build; \
		if [ -f "tmp/dockerfile-hash.txt" ] && sha256sum -c tmp/dockerfile-hash.txt > /dev/null 2>&1; then \
			echo -e "\e[1;36mCached image is up to date\e[0m"; \
		else \
			echo -e "\e[1;36mCached image is different, replacing the cached image\e[0m"; \
			$(MAKE) save; \
			sha256sum Dockerfile > tmp/dockerfile-hash.txt; \
		fi; \
	else \
		echo -e "\e[1;36mNetwork is down, finding the cached image\e[0m"; \
		if [ -f "tmp/image/barn-challenge-2026.tar" ]; then \
			echo -e "\e[1;36mCached image found, loading the image\e[0m"; \
			$(MAKE) load; \
		else \
			echo -e "\e[1;36mCached image not found, checking if the image is already present\e[0m"; \
			if [ -z "$$(docker images -q barn-challenge-2026:latest)" ]; then \
				echo -e "\e[31mImage not found, please build the image when connected to a network\e[0m"; \
				exit 1; \
			else \
				echo -e "\e[1;36mImage found in docker.\e[0m"; \
			fi; \
		fi; \
	fi

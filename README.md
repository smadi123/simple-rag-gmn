# simple-rag-gmn

# Build the backend, frontend and service images
~/projects/simple-rag-gmn/backend$ docker build -t backend-rag .
~/projects/simple-rag-gmn/backend$ docker build -t frontend-rag .
~/projects/simple-rag-gmn/backend$ docker build -t service-rag .

# build & run containers  
~/projects/simple-rag-gmn$ docker-compose up --build 
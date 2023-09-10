# graphdb-image-similarity
Generate Image Recommendation though embeddings with graphdb


docker command 
docker run -it --rm ^
  --publish=7474:7474 --publish=7687:7687 ^
  -e NEO4J_AUTH=neo4j/password ^
  --env NEO4J_PLUGINS='["graph-data-science"]' ^
  neo4j:latest
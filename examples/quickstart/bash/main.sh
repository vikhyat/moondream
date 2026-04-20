# Example cURL Request

## To run locally
curl --location "http://localhost:2020/v1/{endpoint}" \
  --header 'Content-Type: application/json' \
  --data '{
    "image_url": "data:image/jpeg;base64,${cat ../images/frieren.jpg | base64}",
    "stream": false
  }'

## To run on cloud
# curl --location "https://api.moondream.ai/v1/{endpoint}"
#   --header 'X-Moondream-Auth: ${MOONDREAM_API_KEY}' \
#   --header 'Content-Type: application/json' \
#   --data '{
#     "image_url": "data:image/jpeg;base64,${cat ../images/frieren.jpg | base64}",
#     "stream": false
#   }'

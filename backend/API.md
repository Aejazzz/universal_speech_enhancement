# API Documentation

## Endpoints

### `GET /health`

Returns system status.

### `POST /enhance`

Multipart upload:

- `file`: wav/mp3/flac

Response includes:

- routing decision and probabilities
- enhanced audio path
- metrics JSON path
- routing log path
- CSV summary path
- generated plot paths

## OpenAPI

Run backend and open:

- `http://localhost:8000/docs`
- `http://localhost:8000/openapi.json`

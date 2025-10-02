# AI-on-Demand Hugging Face connector
Collects dataset metadata from [Hugging Face](https://huggingface.co) and uploads it to AI-on-Demand.

This package is not intended to be used directly by others, but may serve as an example of how to build a connector for the AI-on-Demand platform.
For more information on how to test this connector locally as a springboard for developing your own connector, reference the [Development](#Development) section below.

### TODO
This package is work in progress.

- [ ] Automatically publish to DockerHub on release
- [ ] Add tests
- [ ] ? Adding back feature for registering the publications

## Installation
You can use the image directly from Docker Hub (TODO) or build it locally.

From Docker Hub: `docker pull aiondemand/huggingface-connector`.

To build a local image:

 - Clone the repository: `git clone https://github.com/aiondemand/huggingface-connector && cd huggingface-connector`
 - Build the image: `docker build -t aiondemand/huggingface-connector -f Dockerfile .`

### Configuring Client Credentials
You will need to configure what server the connector should connect to, as well as the credentials for the client that allow you to upload data.
The connector requires a `config.toml` file with a valid [aiondemand configuration](https://aiondemand.github.io/aiondemand/api/configuration/),
the default configuration can be found in the [`/script/config.prod.toml`](/script/config.prod.toml) file.
You will also need to have the 'Client Secret' for the client, which can be obtained from the keycloak administrator.
The client secret must be provided to the Docker container as an environment variable or in a dotenv file *similar to* [`script/.local.env`](script/.local.env) but named `script/.prod.env`.

Please contact the Keycloak service maintainer to obtain said credentials you need if you are in charge of deploying this Hugging Face connector.

## Running the Connector
For the latest commandline arguments, use `docker run aiondemand/huggingface-connector --help`.

## Development
You can test the connector when running the [metadata catalogue](https://github.com/aiondemand/aiod-rest-api) locally.
The default configurations for this setup can be found in the [`.local.env`](script/.local.env) and [`config.local.toml`](script/config.local.toml) files.

When connecting to the AI-on-Demand test or production server, you will need to a dedicated client registered in the keycloak instance which is connected to the REST API you want to upload data to. 
See [this form]() to apply for a client. The client will need to have a `platform_X` role attached, where `X` is the name of the platform from which you register assets. 
When a client is created, you will need its 'Client ID' and 'Client Secret' and update the relevant configuration and environment files accordingly.

## Disclaimer
This project is not affiliated with Hugging Face in any way.

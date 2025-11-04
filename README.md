# AI-on-Demand Hugging Face connector
Collects dataset metadata from [Hugging Face](https://huggingface.co) and uploads it to AI-on-Demand.

This package is not intended to be used directly by others, but may serve as an example of how to build a connector for the AI-on-Demand platform.
For more information on how to test this connector locally as a springboard for developing your own connector, reference the [Development](#Development) section below.

### TODO
This package is work in progress.

- [ ] Automatically publish to DockerHub on release
- [ ] Add tests

## Installation
You can use the image directly from Docker Hub (TODO) or build it locally.

From Docker Hub: `docker pull aiondemand/openml-connector`.

To build a local image:

 - Clone the repository: `git clone https://github.com/aiondemand/openml-connector && cd openml-connector`
 - Build the image: `docker build -t aiondemand/openml-connector -f Dockerfile .`

### Configuring Client Credentials
You will need to configure what server the connector should connect to, as well as the credentials for the client that allow you to upload data.
The connector requires a `config.toml` file with a valid [aiondemand configuration](https://aiondemand.github.io/aiondemand/api/configuration/),
the default configuration can be found in the [`/script/config.prod.toml`](/script/config.prod.toml) file.
You will also need to have the 'Client Secret' for the client, which can be obtained from the keycloak administrator.
The client secret must be provided to the Docker container as an environment variable or in a dotenv file *similar to* [`script/.local.env`](script/.local.env) but named `script/.prod.env`.

Please contact the Keycloak service maintainer to obtain said credentials you need if you are in charge of deploying this Hugging Face connector.

## Running the Connector
You will need to mount the aiondemand configuration to `/home/appuser/.aiod/config.toml` and provide environment variables directly with `-e` or through mounting the dotfile in `/home/appuser/.aiod/openml/.env`. The [`script/run.sh`](script/run.sh) script provides a convenience that automatically does this. 
It takes one positional argument that has to be `local`, `test`, or `prod` to use the respective files in the `script` folder for configuration.
Any following arguments are interpreted as arguments to the main script.
For the latest commandline arguments, use `docker run aiondemand/openml-connector --help`.
Some example invocations that use the `script/run.sh` script:

 - `script/run.sh local --mode id --value 61 --app-log-level debug` syncs one specific openml dataset, and produces debug logs for the connector only.
 - `script/run.sh test --mode since --value 100 --root-log-level debug` syncs all datasets with identifier 100 or greater (in ascending order).
 - `script/run.sh prod --mode all --root-log-level info` indexes all datasets on OpenML, producing info logs for the connector and all its dependencies (this is the default).

## Development
You can test the connector when running the [metadata catalogue](https://github.com/aiondemand/aiod-rest-api) locally.
The default configurations for this setup can be found in the [`.local.env`](script/.local.env) and [`config.local.toml`](script/config.local.toml) files.

When connecting to the AI-on-Demand test or production server, you will need to a dedicated client registered in the keycloak instance which is connected to the REST API you want to upload data to. 
See [this form]() to apply for a client. The client will need to have a `platform_X` role attached, where `X` is the name of the platform from which you register assets. 
When a client is created, you will need its 'Client ID' and 'Client Secret' and update the relevant configuration and environment files accordingly.

## Disclaimer
This project is not affiliated with OpenML in any way.

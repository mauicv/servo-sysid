import click
from sysid.data_pipeline.generate_actions import generate_actions as _generate_actions
from sysid.data_pipeline.collect_real import collect_responses as _collect_responses
from sysid.data_pipeline.process_data import process_data as _process_data

@click.group()
def cli():
    pass


@cli.command()
def collect_responses():
    _collect_responses()

@cli.command()
def generate_actions():
    _generate_actions()

@cli.command()
def process():
    _process_data()


if __name__ == '__main__':
    cli()
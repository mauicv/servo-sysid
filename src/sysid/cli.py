import click
from sysid.training import train as train_command


@click.group()
def cli():
    pass


@cli.command()
@click.option('--param_set_name', type=str, required=True)
@click.option('--num_generations', type=int, default=100)
@click.option('--population_size', type=int, default=30)
@click.option('--alpha', type=float, default=0.5)
@click.option('--name', type=str, default='experiment')
def train(param_set_name, num_generations, population_size, alpha, name):
    train_command(param_set_name, num_generations, population_size, alpha, name)


if __name__ == '__main__':
    cli()
import multiprocessing
import subprocess
from pathlib import Path
from itertools import product
from collections import defaultdict
import re

import click

_CPU_COUNT = multiprocessing.cpu_count() - 1


def _find_named_configs():
    configs = defaultdict(list)
    for c in Path("configs/").glob("**/*.yaml"):
        parent = str(c.relative_to("configs/").parent)
        name = c.stem
        if parent == ".":
            parent = None
        configs[parent].append(name)
    return configs


_NAMED_CONFIGS = _find_named_configs()


def _get_ingredient_from_mask(mask):
    if "/" in mask:
        return mask.split("/")
    return None, mask


def _validate_config_mask(ctx, param, values):
    for v in values:
        ingredient, _ = _get_ingredient_from_mask(v)
        if ingredient not in _NAMED_CONFIGS:
            raise click.BadParameter(
                f"Invalid ingredient '{ingredient}'. Valid ingredients are: {list(_NAMED_CONFIGS.keys())}"
            )
    return values


def _filter_configs(configs, mask):
    ingredient, mask = _get_ingredient_from_mask(mask)
    regex = re.compile(mask)
    configs[ingredient] = list(filter(regex.search, configs[ingredient]))
    return configs


def work(cmd):
    cmd = cmd.split(" ")
    return subprocess.call(cmd, shell=False)


@click.command()
@click.option("--seeds", default=3, show_default=True, help="How many seeds to run")
@click.option(
    "--cpus",
    default=_CPU_COUNT,
    show_default=True,
    help="How many processes to run in parallel",
)
@click.option(
    "--config-mask",
    "-c",
    multiple=True,
    callback=_validate_config_mask,
    help="Regex mask to filter configs/. Ingredient separator with forward slash \
    '/'. E.g. 'algorithm/rware*'. By default all configs found are used.",
)
def main(seeds, cpus, config_mask):
    pool = multiprocessing.Pool(processes=cpus)

    configs = _NAMED_CONFIGS
    for mask in config_mask:
        configs = _filter_configs(configs, mask)
    configs = [[f"{k}.{i}" if k else str(i) for i in v] for k, v in configs.items()]
    configs += [[f"seed={seed}" for seed in range(seeds)]]

    click.echo("Running following combinations: ")
    click.echo(click.style(" X ", fg="red", bold=True).join([str(s) for s in configs]))

    configs = list(product(*configs))
    if len(configs) == 0:
        click.echo("No valid combinations. Aborted!")
        exit(1)

    click.confirm(
        f"There are {click.style(str(len(configs)), fg='red')} combinations of configurations. Up to {cpus} will run in parallel. Continue?",
        abort=True,
    )

    configs = [
        "python train.py -u with dummy_vecenv=True " + " ".join(c) for c in configs
    ]

    print(pool.map(work, configs))


if __name__ == "__main__":
    main()

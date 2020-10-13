import os

import torch


class ModelSaver:
    """
    Class to save model parameters
    """

    def __init__(self, models_dir="models", run_name="default"):
        self.models_dir = models_dir
        self.run_name = run_name

    def clear_models(self):
        """
        Remove model files in model dir
        """
        if not os.path.isdir(self.models_dir):
            return
        model_dir = os.path.join(self.models_dir, self.run_name)
        if not os.path.isdir(model_dir):
            return
        for f in os.listdir(model_dir):
            f_path = os.path.join(model_dir, f)
            if not os.path.isfile(f_path):
                continue
            os.remove(f_path)

    def save_models(self, alg, extension):
        """
        generate and save networks
        :param model_dir_path: path of model directory
        :param run_name: name of run
        :param alg_name: name of used algorithm
        :param alg: training object of trained algorithm
        :param extension: name extension
        """
        if not os.path.isdir(self.models_dir):
            os.mkdir(self.models_dir)
        model_dir = os.path.join(self.models_dir, self.run_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        for i, agent in enumerate(alg.agents):
            name = "iql_agent%d_params_" % i
            name += extension
            torch.save(agent.model.state_dict(), os.path.join(model_dir, name))

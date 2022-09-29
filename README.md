rf_diffusion README

# Working with arguments (training) & configs (inference)
### JW

The repo is now set up to better syncronise training and inference, but this comes with a few requirements for everyone when developing/exploring training & inference strategies.

Training is governed by `arguments.py`, which is broken up into arguments groups. Some of these pertain to model parameters, loss parameters, diffusion parameters and preprocessing parameters.

`arguments.py` looks a lot like the hydra configs, and it might be a good idea to swap over to configs during training at some point, but for now, these are the (suggested) arguments.py rules:

- All toggleable things have an optional argument in `arguments.py`. If they are important for inference, they should be added to the most appropriate loss group out of 'diff_group', 'preprocess_group', or 'trunk_group'.

- These then get saved into the model checkpoint, which now has a `config_dict` object. `arguments.py` is annoying in that you have to make sure the arguments are correctly passed to the respective `params` dictionary, so be careful.

Inference is governed by hydra configs. These configs will now get the diffuser, preprocess and model parameters from the model checkpoint (if available).

There are several checks in this:

- Backwards compatability should be straightforward, as it's okay for something to be in the config, but not in the checkpoint `config_dict`. There will be a warning thrown to warn the user that this has happened, but as long as the config option matches how the model was trained, this should be okay.
- All items in the checkpoint `config_dict` must be in the inference config. This is to ensure that we never forget to apply a new training feature to inference.
- You can still overwrite configs at inference, even if they're in the model, diffuser or preprocess parameters, but a warning will be thrown. This will allow us to use models outside of the regime in which they were trained, but the warning should hopefully prevent this happening accidentally.

Here is an example of how to add a feature, in a way that won't break things.
E.g. we want to add the option to toggle the output scaling of the SE3 in `Track_module.py`. Currently, it's hard coded at `/ 10.0`, but we want to be able to vary this from `arguments.py`, and correspondingly at inference time.
You would add the argument to `arguments.py` trunk_group, and plumb it in such that it can modify `Track_module.py`. This would then get saved in the model checkpoint. The default of this in the inference config file *must* be what it was originally (`/ 10`). This way, when people use old models at inference time, there will be a warning that 'SE3_scaling' is not in the checkpoint `config_dict`, but the default value will be good (`/ 10'), so inference will proceed and the old model will be used correctly. If you subsequently find that a different SE3_scaling improves training, you can (and should!) change the default in arguments.py. Newer models trained will therefore, by default, train with this improved `SE3_scaling`, and this scaling will be saved in the checkpoint `config_dict`. This will then be auto-loaded into inference config, and hence will also work correctly without user intervention.
The only way this can break old models is if we add a new feature with a default that differs from how RF was originally set up.


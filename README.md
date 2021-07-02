# AppSciTools
This repository stores useful plotting tools and functions that could help an Application Scientist. It could be embedded into multiple projects and iteratively updated to have the most up-to-date tools.
This is not intended to be a module of it's own; rather, it is iteratively loaded as a submodule for each application. By iteratively loading these tools as submodules, we avoid the dependency of a separate module when creating a new application. As a result, sharing applications with all the codes embedded correctly is much easier; rather than requiring multiple dependencies from different directories. 

# Usage
You could add these tools as a submodule to an application that you are working on. To add as a submodule:
`git submodule add git@github.com:alexkchew/AppSciTools.git scripts/AppSciTools`

Then, update submodules after cloning:
`git submodule update --init`

After loading the submodule, you could edit the files and update them. Upon pushing, you could then update all plotting functions across multiple application areas. 

# Description of files
- `plot_tools.py`:
	This code contains useful plotting tools using `matplotlib` module to get the correct figure dimensions


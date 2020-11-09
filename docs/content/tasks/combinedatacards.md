If you are starting with multiple datacards you can use the `CombineDatacards` task to combine them.
You can run this task with:

Let combine automatically choose bin names:

```shell
law run CombineDatacards --version dev --datacards "/path/to/first/card.txt,/path/to/second/card.txt"
```
or use your own bin names:

```shell
law run CombineDatacards --version dev --datacards "first=/path/to/first/card.txt,second=/path/to/second/card.txt"
```

You can pass multiple comma-seperated datacard paths to the `--datacards` cli option.
It also supports globbing, such as:

```shell
law run CombineDatacards --version dev --datacards "/path/to/some/cards/but_only_these*.txt"
```

Relative datacards paths are resolved to your current directory, or, when no file was found, relative to the `datacards_run2` directory in the top level of the repository.
Also, when the path you passed matches a directory, it automatically tries to find a file "datacard.txt" in that directory.
This is meant to simplify the task configuration on the command line as it allows to pass `--datacards bbgg/v0`, provided that the file `repo_path/datacards_run2/bbgg/v0/datacard.txt` exists.

In case you want to give your combined datacard a certain prefix you can use the `--dc-prefix` cli option:

```shell
law run CombineDatacards --version dev --datacards "/path/to/some/cards/but_only_these*.txt" --dc-prefix "my_" --print-status 0
```

Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=my_, hh_model=hh:HHdefault)
|  - LocalFileTarget(path=$DHI_STORE/CombineDatacards/m125.0/model_hh_HHdefault/dev/my_datacard.txt)
|    absent
```

---

**_NOTES_**

As many datacards can make the Task representation unreadable, the input datacard names are hashed and used for the `__repr__`.
In case you pass only 1 datacard to the `--datacards` cli option, this datacard will just be forwarded and nothing happens.

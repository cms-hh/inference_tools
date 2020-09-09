# Combine Datacards

Default: Every law.Task comes with a `--version` parameter, in order to handle multiple inference analysis in parallel.

If you are starting with multiple datacards you can use the `dhi.CombDatacards` task to combine them.
You can run this task with:

```shell
law run dhi.CombDatacards --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt"
```

Note: In case you pass only 1 datacard to the `--input-cards` cli option, this datacard will just be forwarded and nothing happends.

You can pass multiple comma-seperated datacard paths to the `--input-cards` cli option. It also supports globbing, such as:

```shell
law run dhi.CombDatacards --version dev --input-cards "/path/to/some/cards/but_only_these*.txt"
```

In case you want to give your combined datacard a certaint prefix you can use the `--dc-prefix` cli option:

- Normal:

```shell
law run dhi.CombDatacards --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --print-status 0
```
Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of dhi.CombDatacards(version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  datacard: LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/CombDatacards/dev/125/HHdefault/datacard.txt)
|    absent
```

- With prefix `"my_"`:

```shell
law run dhi.CombDatacards --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --dc-prefix "my_" --print-status 0
```
Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of dhi.CombDatacards(version=dev, mass=125, input_cards=, dc_prefix=my_, hh_model=HHdefault, stack_cards=False)
|  datacard: LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/CombDatacards/dev/125/HHdefault/my_datacard.txt)
|    absent
```

By now you might have already noticed that almost all cli options are parsed into the storage path of the outcoming combined datacard.

### Todo

* `--stack-cards` cli option

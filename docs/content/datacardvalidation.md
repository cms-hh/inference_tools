This section will explain how you can validate your datacard.

Check task status:
```shell
law run dhi.ValidateDatacard --input-card /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt --mass 125 --print-status 0
```
Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of dhi.ValidateDatacard(mass=125, input_card=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt, verbosity=1)
|  - LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/ValidateDatacard/125/validation.json)
|    absent
```

Run validation:
```shell
law run dhi.ValidateDatacard --input-card /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt --mass 125
```
Output:
```shell
INFO: Informed scheduler that task   dhi.ValidateDatacard__afs_cern_ch_use_125_1_de72c0d5bd   has status   PENDING
INFO: Done scheduling tasks
INFO: Running Worker with 1 processes
INFO: [pid 19954] Worker Worker(salt=161905714, workers=1, host=lxplus7112.cern.ch, username=<username>, pid=19954) running   dhi.ValidateDatacard(mass=125, input_card=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt, verbosity=1)
ValidateDatacards.py /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt --mass 125 --printLevel 1 --jsonFile /eos/user/<u>/<username>/dhi/store/ValidateDatacard/125/validation.json
[SetFlag] Changing value of flag "check-negative-bins-on-import" from 1 to 0
[SetFlag] Changing value of flag "workspaces-use-clone" from 0 to 1
================================
=======Validation results=======
================================
>>>There were no warnings of type  'up/down templates vary the yield in the same direction'
>>>There were no warnings of type  'up/down templates are identical'
>>>There were no warnings of type  'At least one of the up/down systematic uncertainty templates is empty'
>>>There were  3 warnings of type  'Uncertainty has normalisation effect of more than 10.0%'
>>>There were  34 warnings of type  'Uncertainty probably has no genuine shape effect'
>>>There were no warnings of type 'Empty process'
>>>There were  1 warnings of type  'Bins of the template empty in background'
>>>There were no alerts of type 'Small signal process'
done (took 3.36 seconds)
INFO: [pid 19954] Worker Worker(salt=161905714, workers=1, host=lxplus7112.cern.ch, username=<username>, pid=19954) done      dhi.ValidateDatacard(mass=125, input_card=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt, verbosity=1)
INFO: Informed scheduler that task   dhi.ValidateDatacard__afs_cern_ch_use_125_1_de72c0d5bd   has status   DONE
INFO: Worker Worker(salt=161905714, workers=1, host=lxplus7112.cern.ch, username=<username>, pid=19954) was stopped. Shutting down Keep-Alive thread
INFO:
===== Luigi Execution Summary =====

Scheduled 1 tasks of which:
* 1 ran successfully:
    - 1 dhi.ValidateDatacard(...)

This progress looks :) because there were no failed tasks or missing dependencies

===== Luigi Execution Summary =====
```

You can modify the verbosity level additionally with the `--verbosity` cli option.

In the best case the won't be any warnings. Be careful: empty/negative bins/histograms will lead to `Bogus Norms`!

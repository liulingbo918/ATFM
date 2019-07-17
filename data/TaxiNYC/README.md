**TaxiBJ** consists of the following **Three** datasets:

* NYC2014.h5
* Meteorology.h5
* Holiday.txt

## Flows of Crowds

File names: `NYC2014.h5`, which has two following subsets:

* `date`: a list of timeslots, which is associated the **data**. 
* `data`: a 4D tensor of shape (number_of_timeslots, 2, 15, 5), of which `data[i]` is a 3D tensor of shape (2, 15, 5) at the timeslot `date[i]`, `data[i][0]` is a `15x5` inflow matrix and `data[i][1]` is a `15x5` outflow matrix. 


## Meteorology

File name: `Meteorology.h5`, which has four following subsets:

* `date`: a list of timeslots, which is associated the following kinds of data. 
* `Temperature`: a list of continuous value, of which the `i^{th}` value is `temperature` at the timeslot `date[i]`.
* `WindSpeed`: a list of continuous value, of which the `i^{th}` value is `wind speed` at the timeslot `date[i]`. 
* `Weather`: a 2D matrix, each of which is a one-hot vector (`dim=17`), showing one of the following weather types: 
```
{'Cloudy': 0, 'Cloudy / Windy': 1, 'Fair': 2, 'Fair / Windy': 3, 'Fog': 4, 'Haze': 5, 'Heavy Rain': 6, 'Heavy Rain / Windy': 7, 'Light Rain': 8, 'Light Snow': 9, 'Mostly Cloudy': 10, 'Partly Cloudy': 11, 'Rain': 12, 'Rain / Windy': 13, 'Unknown': 14, 'Unknown Precipitation': 15, 'Wintry Mix': 16}
```

## Holiday

File name: `BJ_Holiday.txt`, which inclues a list of the holidays (and adjacent weekends) of Beijing. 

Each line a holiday with the data format [yyyy][mm][dd]. For example, `20150601` is `June 1st, 2015`. 

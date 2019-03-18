// Currency
SELECT
currency_unit,
COUNT(1) as c
 	FROM  `bigquery-public-data.world_bank_health_population.country_summary` 
 	 	GROUP BY currency_unit
 		ORDER BY c
 			DESC LIMIT 100;

// Examples
SELECT
 	*
 	FROM `bigquery-public-data.world_bank_health_population.country_summary`
 		LIMIT 100;      

// Examples table 2
SELECT
 	*
 		FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`
 			WHERE
 				country_code = "USA"
 				AND year = 2017
 					LIMIT 100;    

// Top indicators
SELECT
 	indicator_name,
 	COUNT(1) as c
 		FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population` 
 		WHERE
 			year = 2017
 				GROUP BY indicator_name
 				ORDER BY c DESC
 				LIMIT 100;          

// Immunization vs mortality rate
SELECT
  indicator_name as indicator,
  country_name as country,
  value
    FROM `bigquery-public-data.world_bank_health_population.health_nutrition_population`
      WHERE
        indicator_name IN ("Immunization, DPT (% of children ages 12-23 months)", "Mortality rate, infant, female (per 1,000 live births)")
        AND year = 2017
        

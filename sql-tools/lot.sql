SET ANSI_NULLS on
set QUOTED_IDENTIFIER  ON
select     top 3             
	   CONVERT(VARCHAR(10),CAST(CAST(Aces.dbo.STUDIES_TABLE.date AS Datetime) AS Date)) AS Date,                      
           CONVERT(VARCHAR(8),CAST(CAST(Aces.dbo.STUDIES_TABLE.START_TIME AS Datetime) AS Time)) AS Carto_Start,
           CONVERT(VARCHAR(9),Points.First_Point) AS map_start,           
           CONVERT(VARCHAR(9),RFTime.First_RF) AS rf_start,
           CONVERT(VARCHAR(9),RFTime.Last_RF) AS rf_end,
	   CONVERT(VARCHAR(9),Points2.Last_Point) AS validation_end,
	   CONVERT(VARCHAR(9),RFTime.TotalRF) AS TotalRF,
	   CONVERT(VARCHAR(9),RFTime.numRF) AS numRF
from       Aces.dbo.STUDIES_TABLE
LEFT JOIN       Aces.dbo.SETUP_TABLE
ON         Aces.dbo.SETUP_TABLE.SETUP_IDX = Aces.dbo.STUDIES_TABLE.SETUP_IDX
LEFT JOIN	 Aces.dbo.PATIENTS_TABLE
ON	   Aces.dbo.PATIENTS_TABLE.IDX = Aces.dbo.STUDIES_TABLE.PATIENT_IDX
LEFT JOIN
( select   Aces.dbo.STUDIES_TABLE.STUDY_IDX,    
           CAST(CAST(MIN(Aces.dbo.POINTS_TABLE.ACQUISITION_TIME) AS Datetime)AS Time(0)) AS First_Point
  from     Aces.dbo.STUDIES_TABLE,
           Aces.dbo.MAPS_TABLE,
	       Aces.dbo.POINTS_TABLE
  where    Aces.dbo.STUDIES_TABLE.STUDY_IDX = Aces.dbo.MAPS_TABLE.STUDY_IDX
  and      Aces.dbo.MAPS_TABLE.MAP_IDX = Aces.dbo.POINTS_TABLE.MAP_IDX
  group by Aces.dbo.STUDIES_TABLE.STUDY_IDX
) AS Points
ON Points.STUDY_IDX = Aces.dbo.STUDIES_TABLE.STUDY_IDX
LEFT JOIN
( select   Aces.dbo.STUDIES_TABLE.STUDY_IDX,    
           CAST(CAST(MAX(Aces.dbo.POINTS_TABLE.ACQUISITION_TIME) AS Datetime)AS Time(0)) AS Last_Point
  from     Aces.dbo.STUDIES_TABLE,
           Aces.dbo.MAPS_TABLE,
	       Aces.dbo.POINTS_TABLE
  where    Aces.dbo.STUDIES_TABLE.STUDY_IDX = Aces.dbo.MAPS_TABLE.STUDY_IDX
  and      Aces.dbo.MAPS_TABLE.MAP_IDX = Aces.dbo.POINTS_TABLE.MAP_IDX
  group by Aces.dbo.STUDIES_TABLE.STUDY_IDX
) AS Points2
ON Points2.STUDY_IDX = Aces.dbo.STUDIES_TABLE.STUDY_IDX
LEFT JOIN
( select   Aces.dbo.STUDIES_TABLE.STUDY_IDX, 
           CAST(CAST(MIN(Aces.dbo.RF_TABLE.ABLATION_START_TIME) AS Datetime)AS Time(0)) AS First_RF,
       	   CAST(CAST(MAX(Aces.dbo.RF_TABLE.ABLATION_START_TIME) AS Datetime)AS Time(0)) AS Last_RF,
           SUM(Aces.dbo.RF_TABLE.RF_TIME/1000)/60 AS TotalRF,
	   AVG(Aces.dbo.RF_TABLE.RF_TIME)/1000 AS MeanRF,
	   COUNT(Aces.dbo.RF_TABLE.RF_TIME) AS numRF
  from     Aces.dbo.STUDIES_TABLE,
           Aces.dbo.MAPS_TABLE,
	       Aces.dbo.RF_TABLE
  where    Aces.dbo.STUDIES_TABLE.STUDY_IDX = Aces.dbo.MAPS_TABLE.STUDY_IDX
  and	   Aces.dbo.RF_TABLE.MAP_IDX = Aces.dbo.MAPS_TABLE.MAP_IDX
  group by Aces.dbo.STUDIES_TABLE.STUDY_IDX
) AS RFTime
ON RFTime.STUDY_IDX = Aces.dbo.STUDIES_TABLE.STUDY_IDX
LEFT JOIN
(
SELECT DISTINCT(Aces.dbo.STUDIES_TABLE.STUDY_IDX), CONFIG_MAIN_TABLE.PROCEDURE_IDX AS ptype,
			STUFF((
          			SELECT ',' + Aces.dbo.CONFIG_CONNECTOR_TABLE.CATHETER_NAME
          			FROM  Aces.dbo.CONFIG_CONNECTOR_TABLE
          			WHERE Aces.dbo.CONFIG_MAIN_TABLE.CONFIG_IDX = Aces.dbo.CONFIG_CONNECTOR_TABLE.CONFIG_IDX
          			AND Aces.dbo.CONFIG_CONNECTOR_TABLE.IS_IN_CONNECTOR_SETUP =1
          			FOR XML PATH(''), TYPE).value('.', 'NVARCHAR(MAX)'
						 ), 1, 1, '') as names
FROM Aces.dbo.STUDIES_TABLE,
     Aces.dbo.CONFIG_MAIN_TABLE,
     Aces.dbo.CONFIG_CONNECTOR_TABLE
WHERE
 	Aces.dbo.STUDIES_TABLE.SETUP_IDX = Aces.dbo.CONFIG_MAIN_TABLE.SETUP_IDX
AND
 	Aces.dbo.CONFIG_MAIN_TABLE.CONFIG_IDX = Aces.dbo.CONFIG_CONNECTOR_TABLE.CONFIG_IDX
) AS Cath
ON Cath.STUDY_IDX = Aces.dbo.STUDIES_TABLE.STUDY_IDX
LEFT JOIN
( select   Aces.dbo.STUDIES_TABLE.STUDY_IDX, 
           AVG(Aces.dbo.RF_TABLE.AVERAGE_CONTACT_FORCE) AS CF_MEAN,
           STDEV(Aces.dbo.RF_TABLE.AVERAGE_CONTACT_FORCE) AS CF_STD
  from     Aces.dbo.STUDIES_TABLE,
           Aces.dbo.MAPS_TABLE,
	   Aces.dbo.RF_TABLE
  where    Aces.dbo.STUDIES_TABLE.STUDY_IDX = Aces.dbo.MAPS_TABLE.STUDY_IDX
  and	   Aces.dbo.RF_TABLE.MAP_IDX = Aces.dbo.MAPS_TABLE.MAP_IDX
  AND    Aces.dbo.RF_TABLE.AVERAGE_CONTACT_FORCE > 0
  group by Aces.dbo.STUDIES_TABLE.STUDY_IDX

) AS RFForce
ON RFForce.STUDY_IDX = Aces.dbo.STUDIES_TABLE.STUDY_IDX
LEFT JOIN  Aces.dbo.PROCEDURES_TABLE
on  Aces.dbo.PROCEDURES_TABLE.PROCEDURE_IDX = Cath.ptype
ORDER BY Date DESC


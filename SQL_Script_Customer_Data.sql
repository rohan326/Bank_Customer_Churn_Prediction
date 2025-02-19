-- creating the database and loading the csv file of the customer data

create database customer_data;
use customer_data;

-- verifying the tables
show tables;

-- The table contains whitespaces so renaming the table name 
ALTER TABLE `bank customer churn prediction` rename to bank_customer_churn_prediction;

-- retreiving the data
select * from bank_customer_churn_prediction;

-- Data Cleaning

-- Removing the duplicates
select distinct * from bank_customer_churn_prediction;
-- looking for null values

Select count(* ) as null_count from bank_customer_churn_prediction
where customer_id is null or
	credit_score is null or
	country is null or
	gender is null or 
	age is null or 
	tenure is null or 
	balance is null or 
	products_number is null or
    credit_card is null or
    active_member is null or
    estimated_salary is null or
    churn;

-- creating a duplicate table

create table copy
like bank_customer_churn_prediction;
select * from copy;

insert copy
select * from bank_customer_churn_prediction;

-- Removing Duplicates
select * from copy;
with duplicate_cte as
(
select* ,
row_number() over(partition by customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, churn) as row_num
from copy
)
select * from duplicate_cte
where row_num > 1;

-- standardizing the data

select customer_id , trim(customer_id)
from copy;
update copy 
set customer_id = trim(customer_id);

select credit_score,
row_number() over(partition_by order by credit_score) as row_num_credit from copy;


with credit_dupe_cte as (
SELECT credit_score,
       row_number() over (partition by credit_score) as row_num_credit
from copy
)
select * from credit_dupe_cte
where row_num_credit > 1;


select distinct country from copy;

# Exploratory data analysis
select * from copy;

select country,  max(credit_score), min(credit_score), avg(credit_score)
from copy
group by country;

select distinct country from copy;
select max(age), min(age), avg(age) from copy;
select max(balance), min(balance), avg(balance) from copy;
select max(estimated_salary), min(estimated_salary), avg(estimated_salary) from copy;






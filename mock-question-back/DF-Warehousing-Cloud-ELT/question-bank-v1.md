# Interview Questions: Week 4 - Cloud Data, Warehousing, and ETL/ELT

## Beginner (Foundational)

### Q1: What are the three main cloud service models?
**Keywords:** IaaS, PaaS, SaaS, Infrastructure, Platform, Software
<details>
<summary>Click to Reveal Answer</summary>

| Model | Name | What You Manage | Provider Manages |
|-------|------|-----------------|------------------|
| **IaaS** | Infrastructure as a Service | Apps, Data, Runtime, OS | VMs, Storage, Network |
| **PaaS** | Platform as a Service | Applications, Data | Runtime, OS, Infrastructure |
| **SaaS** | Software as a Service | Just use the app | Everything |

**Examples:**
- IaaS: AWS EC2, Azure VMs, Google Compute Engine
- PaaS: BigQuery, AWS RDS, Cloud SQL
- SaaS: Salesforce, Google Workspace, Tableau Online

Higher abstraction = less control but less management overhead.
</details>

---

### Q2: What are the five Vs of big data?
**Keywords:** Volume, Velocity, Variety, Veracity, Value
<details>
<summary>Click to Reveal Answer</summary>

| V | Description |
|---|-------------|
| **Volume** | Amount of data (terabytes to petabytes) |
| **Velocity** | Speed of data generation and processing |
| **Variety** | Different data types (structured, semi-structured, unstructured) |
| **Veracity** | Accuracy and trustworthiness of data |
| **Value** | Business value derived from data |

These characteristics define what makes data "big" and the challenges in processing it.
</details>

---

### Q3: What is the difference between OLTP and OLAP systems?
**Keywords:** Transactional, Analytical, Current, Historical, Fast Writes, Complex Reads
<details>
<summary>Click to Reveal Answer</summary>

| Aspect | OLTP | OLAP |
|--------|------|------|
| Purpose | Run operations | Analyze trends |
| Users | Clerks, apps | Analysts, managers |
| Data focus | Current state | Historical trends |
| Operations | INSERT, UPDATE, DELETE | SELECT (read-heavy) |
| Response | Milliseconds | Seconds to minutes |
| Schema | Normalized (3NF) | Denormalized (Star) |
| Data volume | Gigabytes | Terabytes+ |

Most organizations need both: OLTP for operations, OLAP for analytics.
</details>

---

### Q4: What is a data warehouse and how does it differ from a transactional database?
**Keywords:** Analytical, Historical, Read-Optimized, Aggregations, Reporting
<details>
<summary>Click to Reveal Answer</summary>

A **data warehouse** is a centralized repository optimized for analytical queries and reporting.

| Aspect | Data Warehouse | Transactional DB |
|--------|----------------|------------------|
| Purpose | Analytics, BI | Day-to-day operations |
| Optimized for | Read queries | Frequent writes |
| Schema | Denormalized | Normalized |
| Data | Historical, integrated | Current, operational |
| Users | Analysts | Applications |

Examples: BigQuery, Snowflake, Redshift vs. PostgreSQL, MySQL, Oracle
</details>

---

### Q5: What is a star schema and what are its components?
**Keywords:** Fact Table, Dimension Tables, Measures, Foreign Keys, Center
<details>
<summary>Click to Reveal Answer</summary>

A star schema consists of one central **fact table** surrounded by **dimension tables**, forming a star shape:

**Fact Table (Center):**
- Contains measurable metrics (sales amount, quantity)
- Has foreign keys to all dimensions
- Usually the largest table

**Dimension Tables (Points):**
- Contain descriptive attributes (customer name, product category)
- Have primary keys matching fact FKs
- Provide context for analysis

```
     dim_date     dim_customer
          \           /
           \         /
          +--fact_sales--+
           /           \
          /             \
     dim_product    dim_store
```
</details>

---

### Q6: What is the difference between structured, semi-structured, and unstructured data?
**Keywords:** Schema, Tables, JSON, XML, Text, Images
<details>
<summary>Click to Reveal Answer</summary>

| Type | Structure | Examples |
|------|-----------|----------|
| **Structured** | Fixed schema, rows/columns | Database tables, CSV, Excel |
| **Semi-structured** | Flexible schema, self-describing | JSON, XML, Parquet |
| **Unstructured** | No predefined schema | Text, images, videos, logs |

Storage considerations:
- Structured: Relational databases
- Semi-structured: NoSQL, Data lakes
- Unstructured: Object storage, Data lakes
</details>

---

### Q7: What is the difference between ETL and ELT?
**Keywords:** Extract, Transform, Load, Sequence, Where Transformation Happens
<details>
<summary>Click to Reveal Answer</summary>

| Aspect | ETL | ELT |
|--------|-----|-----|
| Full name | Extract, Transform, Load | Extract, Load, Transform |
| Transform location | External ETL server | Inside data warehouse |
| Raw data in target | No (transformed only) | Yes (preserved) |
| Flexibility | Less (fixed transforms) | More (re-transform anytime) |
| Best for | Legacy systems, compliance | Cloud warehouses |

```
ETL: Source -> Extract -> Transform -> Load -> Warehouse
ELT: Source -> Extract -> Load -> Transform -> Warehouse
```

ELT is becoming standard for modern cloud-native architectures.
</details>

---

### Q8: What are common data file formats and their use cases?
**Keywords:** CSV, JSON, Parquet, Avro, ORC
<details>
<summary>Click to Reveal Answer</summary>

| Format | Type | Best For |
|--------|------|----------|
| **CSV** | Row-based, text | Simple data exchange, compatibility |
| **JSON** | Semi-structured | APIs, web data, flexibility |
| **Parquet** | Columnar, binary | Analytics, BigQuery, Spark |
| **Avro** | Row-based, binary | Streaming, schema evolution |
| **ORC** | Columnar, binary | Hive, Hadoop analytics |

For analytics workloads, columnar formats (Parquet, ORC) provide better compression and query performance.
</details>

---

### Q9: What is BigQuery and what type of service model is it?
**Keywords:** Serverless, Data Warehouse, PaaS, Google Cloud, SQL
<details>
<summary>Click to Reveal Answer</summary>

**BigQuery** is Google Cloud's fully-managed, serverless data warehouse.

**Service Model:** PaaS (Platform as a Service)

**Key Features:**
- No servers to provision or manage
- Scales automatically
- Pay for queries executed and storage used
- Standard SQL interface
- Supports partitioning and clustering
- Integrated with GCP ecosystem

You focus on writing SQL queries; Google handles all infrastructure.
</details>

---

### Q10: What is denormalization and why is it used in data warehouses?
**Keywords:** Redundancy, Read Performance, Fewer Joins, Star Schema
<details>
<summary>Click to Reveal Answer</summary>

**Denormalization** intentionally adds redundancy to improve read performance.

| Aspect | Normalized | Denormalized |
|--------|------------|--------------|
| Redundancy | Minimal | Intentional |
| Joins | Many, complex | Few, simple |
| Read speed | Slower | Faster |
| Write speed | Faster | Slower |
| Best for | OLTP | OLAP |

Example: Storing customer city in both customer dimension AND fact table to avoid joins.

Data warehouses use denormalization (star schemas) because they prioritize read performance over write efficiency.
</details>

---

## Intermediate (Application)

### Q11: How would you choose between IaaS and PaaS for a data analytics workload?
**Hint:** Think about control vs. management overhead.
<details>
<summary>Click to Reveal Answer</summary>

**Choose IaaS when:**
- Need full control over OS and software stack
- Running custom/legacy software
- Specific compliance requirements
- Migrating existing on-premises workloads

**Choose PaaS when:**
- Want to focus on data, not infrastructure
- Team lacks infrastructure expertise
- Need rapid scaling
- Using standard analytics tools (SQL queries)

**Example Decision:**
- Running custom Spark cluster: IaaS (Compute Engine + self-managed)
- Running SQL analytics: PaaS (BigQuery - no infrastructure management)

For most analytics use cases, PaaS provides better time-to-value.
</details>

---

### Q12: Explain the difference between a data warehouse, data lake, and data lakehouse.
**Hint:** Think about structure, processing, and use cases.
<details>
<summary>Click to Reveal Answer</summary>

| Aspect | Data Warehouse | Data Lake | Data Lakehouse |
|--------|----------------|-----------|----------------|
| Data type | Structured | All types | All types |
| Schema | Schema-on-write | Schema-on-read | Both |
| Format | Proprietary | Open (Parquet, JSON) | Open |
| Users | Analysts, BI | Data scientists, engineers | All |
| Cost | Higher | Lower (storage) | Moderate |
| Governance | Strong | Weak | Strong |

**Lakehouse** combines the best of both: open formats of data lakes with ACID transactions and governance of warehouses.
</details>

---

### Q13: What is a surrogate key and why is it used in dimensional modeling?
**Hint:** Think about stability and history tracking.
<details>
<summary>Click to Reveal Answer</summary>

A **surrogate key** is a system-generated, meaningless integer that uniquely identifies dimension rows.

**Why use surrogate keys:**
1. **Stability**: Natural keys can change (email, product SKU)
2. **History**: Enable SCD Type 2 (multiple rows per entity)
3. **Performance**: Integer joins are faster
4. **Integration**: Unify different source systems

```sql
dim_customer
| customer_key | customer_id | name  |
|--------------|-------------|-------|
| 1            | CUST-001    | Alice |  -- Surrogate key: 1
| 2            | CUST-001    | Alice |  -- Same customer, new version
```

Natural key (`customer_id`) identifies the business entity; surrogate key (`customer_key`) identifies the specific row.
</details>

---

### Q14: Describe SCD Type 1 vs. SCD Type 2 for handling dimension changes.
**Hint:** Think about history preservation.
<details>
<summary>Click to Reveal Answer</summary>

**SCD Type 1 (Overwrite):**
- Replace old value with new value
- No history preserved
- Simple to implement

```sql
-- Before: Alice in LA
-- After: Alice in NYC (LA is lost forever)
UPDATE dim_customer SET city = 'NYC' WHERE id = 1;
```

**SCD Type 2 (Add New Row):**
- Create new row for each change
- Full history preserved
- Requires date ranges and current flags

```sql
| key | name  | city | start_date | end_date   | current |
|-----|-------|------|------------|------------|---------|
| 1   | Alice | LA   | 2020-01-01 | 2023-06-30 | N       |
| 2   | Alice | NYC  | 2023-07-01 | 9999-12-31 | Y       |
```

Use Type 1 for corrections; Type 2 for historical accuracy.
</details>

---

### Q15: How do partitioning and clustering improve BigQuery performance?
**Hint:** Think about reducing data scanned.
<details>
<summary>Click to Reveal Answer</summary>

**Partitioning:** Divides table into segments based on a column (usually date).

```sql
CREATE TABLE sales
PARTITION BY DATE(order_date);

-- Query only scans December 2023 partition
SELECT * FROM sales WHERE order_date = '2023-12-15';
```

**Clustering:** Sorts data within partitions by specified columns.

```sql
CREATE TABLE sales
PARTITION BY order_date
CLUSTER BY customer_id, product_id;
```

**Benefits:**
- Partitioning: Skip irrelevant partitions entirely
- Clustering: Efficient filtering within partitions
- Both: Reduce data scanned = faster queries + lower cost
</details>

---

### Q16: What are conformed dimensions and why are they important?
**Hint:** Think about consistency across fact tables.
<details>
<summary>Click to Reveal Answer</summary>

**Conformed dimensions** are dimension tables shared identically across multiple fact tables.

```
fact_sales ----+
               |---> dim_date (shared/conformed)
fact_inventory-+

fact_sales ----+
               |---> dim_product (shared/conformed)
fact_returns---+
```

**Why important:**
1. **Consistency**: Same definitions everywhere
2. **Drill-across**: Query multiple fact tables together
3. **Single source of truth**: One definition for "customer" or "date"
4. **Maintainability**: Update once, apply everywhere

Without conformed dimensions, "Q4 2023" might mean different things in different reports.
</details>

---

### Q17: When would you choose ETL over ELT and vice versa?
**Hint:** Think about compliance, infrastructure, and team skills.
<details>
<summary>Click to Reveal Answer</summary>

**Choose ETL when:**
- Regulatory requirement to filter sensitive data before storage
- Legacy systems with established ETL tools (Informatica, SSIS)
- Complex transformations suit specialized tools
- Small datasets where ETL overhead is minimal

**Choose ELT when:**
- Using cloud warehouses (BigQuery, Snowflake, Redshift)
- Need flexibility to re-transform data later
- Want to preserve raw data for audit/re-processing
- Team has strong SQL skills
- Modern, greenfield architecture

**Hybrid approach:** Light pre-processing (PII masking) + ELT for main transformations.
</details>

---

## Advanced (Deep Dive)

### Q18: Design a star schema for an e-commerce order tracking system. What fact and dimension tables would you create?
<details>
<summary>Click to Reveal Answer</summary>

**Fact Table: `fact_orders`**
```sql
CREATE TABLE fact_orders (
    order_key INT64,           -- Surrogate key
    date_key INT64,            -- FK to dim_date
    customer_key INT64,        -- FK to dim_customer
    product_key INT64,         -- FK to dim_product
    store_key INT64,           -- FK to dim_store (or channel)
    promotion_key INT64,       -- FK to dim_promotion
    -- Measures
    quantity INT64,
    unit_price NUMERIC(10,2),
    discount_amount NUMERIC(10,2),
    total_amount NUMERIC(10,2),
    shipping_cost NUMERIC(10,2)
);
```

**Dimension Tables:**

| Dimension | Key Attributes |
|-----------|----------------|
| `dim_date` | date_key, full_date, day_name, month, quarter, year, is_holiday |
| `dim_customer` | customer_key, name, email, segment, city, state, country |
| `dim_product` | product_key, sku, name, category, subcategory, brand |
| `dim_store` | store_key, store_name, type (online/retail), region |
| `dim_promotion` | promo_key, promo_name, type, discount_percent |

This enables queries like "Total sales by month by product category" with simple joins.
</details>

---

### Q19: Explain SCD Type 6 and when you would use it over simpler SCD types.
<details>
<summary>Click to Reveal Answer</summary>

**SCD Type 6** (also called "Hybrid") combines Types 1, 2, and 3:

```sql
| key | name  | current_city | history_city | start    | end      | current |
|-----|-------|--------------|--------------|----------|----------|---------|
| 1   | Alice | New York     | Los Angeles  | 2020-01  | 2023-06  | N       |
| 2   | Alice | New York     | New York     | 2023-07  | 9999-12  | Y       |
```

**Components:**
- **Type 2**: Multiple rows with date ranges (full history)
- **Type 3**: Previous value column on each row
- **Type 1**: Current value overwritten in ALL rows

**When to use Type 6:**
- Need complete history (Type 2 requirement)
- Also want current value accessible on historical rows
- Complex reporting needs both perspectives
- Willing to accept higher complexity and storage

**Trade-off:** More ETL complexity but maximum analytical flexibility.
</details>

---

### Q20: How would you implement an ELT pipeline in BigQuery that loads raw data, stages it, and transforms it into a star schema?
<details>
<summary>Click to Reveal Answer</summary>

**Step 1: Extract and Load Raw Data**
```sql
-- Load raw JSON from Cloud Storage
CREATE OR REPLACE EXTERNAL TABLE raw.orders_ext
OPTIONS (
    format = 'JSON',
    uris = ['gs://bucket/orders/*.json']
);

-- Materialize to native table
CREATE TABLE staging.orders_raw AS
SELECT * FROM raw.orders_ext;
```

**Step 2: Stage and Clean**
```sql
CREATE TABLE staging.orders_cleaned AS
SELECT
    order_id,
    PARSE_DATE('%Y-%m-%d', order_date) AS order_date,
    CAST(customer_id AS INT64) AS customer_id,
    CAST(product_id AS INT64) AS product_id,
    CAST(quantity AS INT64) AS quantity,
    CAST(total AS NUMERIC) AS total
FROM staging.orders_raw
WHERE order_id IS NOT NULL;
```

**Step 3: Transform to Star Schema**
```sql
-- Load dimension (with SCD Type 1)
MERGE warehouse.dim_customer AS target
USING staging.customers_cleaned AS source
ON target.customer_id = source.customer_id
WHEN MATCHED THEN UPDATE SET name = source.name, city = source.city
WHEN NOT MATCHED THEN INSERT (...);

-- Load fact with dimension keys
INSERT INTO warehouse.fact_orders
SELECT
    GENERATE_UUID() AS order_key,
    FORMAT_DATE('%Y%m%d', o.order_date) AS date_key,
    c.customer_key,
    p.product_key,
    o.quantity,
    o.total
FROM staging.orders_cleaned o
JOIN warehouse.dim_customer c ON o.customer_id = c.customer_id
JOIN warehouse.dim_product p ON o.product_id = p.product_id;
```

This preserves raw data for re-processing while building analytics-ready star schema.
</details>

---

### Q21: Compare the cost and performance trade-offs of storing 1TB of analytics data in Parquet format on Cloud Storage vs. loading it directly into BigQuery.
<details>
<summary>Click to Reveal Answer</summary>

**Cloud Storage + External Table:**

| Aspect | Details |
|--------|---------|
| Storage cost | ~$20/month (standard class) |
| Query cost | Pay for data scanned (no BigQuery optimization) |
| Performance | Slower (no clustering/caching) |
| Best for | Rarely queried archive data |

**Native BigQuery Table:**

| Aspect | Details |
|--------|---------|
| Storage cost | ~$20/month (same as GCS) |
| Query cost | Optimized scans with partitioning/clustering |
| Performance | Fast (columnar, cached, optimized) |
| Best for | Frequently queried analytics data |

**Recommendations:**
- Hot data (queried daily): Native BigQuery with partitioning/clustering
- Warm data (queried weekly): Native BigQuery
- Cold data (rarely queried): GCS with external table or load-on-demand
- Archive (compliance): GCS Coldline/Archive class

**Key insight:** BigQuery storage costs are competitive with GCS, but performance is dramatically better for native tables.
</details>

---

## Additional Data Modeling and Warehousing Questions

### Q22: What are the three levels of data modeling and who is the audience for each?
**Keywords:** Conceptual, Logical, Physical, Business, Technical
<details>
<summary>Click to Reveal Answer</summary>

| Level | Focus | Audience | Includes |
|-------|-------|----------|----------|
| **Conceptual** | What (business concepts) | Executives, business stakeholders | Major entities, high-level relationships, no attributes |
| **Logical** | How (structure) | Analysts, architects | All entities, attributes, keys, cardinality, normalized |
| **Physical** | Implementation | DBAs, developers | Table names, data types, indexes, partitions |

**Progression:**
```
Business Requirements -> Conceptual -> Logical -> Physical -> Database
```

Each level adds more technical detail as you move from business understanding to implementation.
</details>

---

### Q23: What is the "grain" in dimensional modeling and why is it critical?
**Keywords:** Level of Detail, Atomic, One Row Represents, Aggregation
<details>
<summary>Click to Reveal Answer</summary>

The **grain** defines what a single row in a fact table represents - the level of detail captured.

**Examples:**
| Fact Table | Grain |
|------------|-------|
| Sales | One row per line item on a receipt |
| Daily Inventory | One row per product per store per day |
| Monthly Summary | One row per customer per month |

**Why it's critical:**
- Determines what questions can be answered
- You can always aggregate UP from atomic grain
- You CANNOT disaggregate DOWN

**Rule:** Choose the most atomic (finest) grain possible. Store transaction-level data, then aggregate for reports.

```
Atomic grain: Individual sales transactions
  -> Can aggregate to: Daily, weekly, monthly, by product, by customer
```
</details>

---

### Q24: What are the three types of facts (additive, semi-additive, non-additive)?
**Keywords:** Measures, Summable, Aggregation, Metrics
<details>
<summary>Click to Reveal Answer</summary>

| Type | Can Sum Across | Examples |
|------|----------------|----------|
| **Additive** | All dimensions | Revenue, Quantity, Cost |
| **Semi-additive** | Some dimensions (not time) | Account Balance, Inventory Level |
| **Non-additive** | No dimensions | Ratios, Percentages, Averages |

**Example:**
```sql
-- Additive: Sum revenue across any dimension
SELECT region, SUM(revenue) FROM fact_sales GROUP BY region;

-- Semi-additive: Average balance across time, not sum
SELECT customer, AVG(balance) FROM fact_account_balance GROUP BY customer;

-- Non-additive: Cannot sum percentages
SELECT product, profit_margin FROM fact_sales;  -- Don't SUM this!
```

Understanding fact types prevents calculation errors in reports.
</details>

---

### Q25: What is a snowflake schema and how does it differ from a star schema?
**Keywords:** Normalized Dimensions, More Joins, Less Redundancy, Sub-Dimensions
<details>
<summary>Click to Reveal Answer</summary>

A **snowflake schema** normalizes dimension tables into multiple related sub-tables.

| Aspect | Star Schema | Snowflake Schema |
|--------|-------------|------------------|
| Dimensions | Flat (denormalized) | Normalized (multiple tables) |
| Joins | Fewer | More |
| Query speed | Faster | Slower |
| Storage | More (redundant) | Less |
| Maintenance | More updates | Fewer updates |

**Example:**
```
Star:      fact_sales -> dim_product (contains category, brand inline)

Snowflake: fact_sales -> dim_product -> dim_category
                                     -> dim_brand
```

**When to use snowflake:**
- Storage is a major concern
- Deep, stable hierarchies
- Dimension attributes change frequently

**Modern guidance:** Star schemas are preferred in cloud warehouses where storage is cheap.
</details>

---

### Q26: What is a degenerate dimension? Give an example.
**Keywords:** No Dimension Table, Stored in Fact, Transaction ID, Invoice Number
<details>
<summary>Click to Reveal Answer</summary>

A **degenerate dimension** is a dimension value stored directly in the fact table with no corresponding dimension table.

**Example:**
```sql
CREATE TABLE fact_sales (
    date_key INT64,
    customer_key INT64,
    product_key INT64,
    invoice_number STRING,  -- Degenerate dimension (no dim_invoice table)
    quantity INT64,
    amount NUMERIC
);
```

**When to use:**
- Transaction identifiers (invoice, order, receipt numbers)
- Values with no additional attributes
- Unique IDs that only serve as grouping keys

**Why not create a dimension table?**
```sql
-- This would be wasteful:
dim_invoice
| invoice_number |  <- Only column, no other attributes!
|----------------|
| INV-001        |
```
</details>

---

### Q27: What is a junk dimension and when would you use one?
**Keywords:** Flags, Indicators, Low Cardinality, Combined Attributes
<details>
<summary>Click to Reveal Answer</summary>

A **junk dimension** combines miscellaneous low-cardinality flags and indicators that do not belong in other dimensions.

**Before (flags in fact table):**
```sql
fact_sales
| payment_type | is_sale | is_return | promo_code |
|--------------|---------|-----------|------------|
| Credit       | Y       | N         | None       |
```

**After (junk dimension):**
```sql
dim_transaction_type
| type_key | payment_type | is_sale | is_return | promo_code |
|----------|--------------|---------|-----------|------------|
| 1        | Credit       | Y       | N         | None       |
| 2        | Credit       | Y       | N         | 10OFF      |
| 3        | Debit        | Y       | N         | None       |

fact_sales
| type_key | quantity | amount |  <- Single FK replaces multiple columns
```

**Benefits:**
- Cleaner fact table
- Pre-defined valid combinations
- Easier to add new flags
</details>

---

### Q28: What is a role-playing dimension? Give an example with dates.
**Keywords:** Same Dimension Multiple Times, Different Roles, Aliases
<details>
<summary>Click to Reveal Answer</summary>

A **role-playing dimension** is a single dimension table used multiple times in a fact table, each time with a different meaning.

**Example: Order fact with multiple dates**
```sql
fact_order
| order_key | order_date_key | ship_date_key | cancel_date_key |
|-----------|----------------|---------------|-----------------|
| 1         | 20240115       | 20240118      | NULL            |
| 2         | 20240120       | NULL          | 20240121        |
```

All three date keys reference the SAME `dim_date` table.

**Query with aliases:**
```sql
SELECT
    o.order_key,
    od.full_date AS order_date,
    sd.full_date AS ship_date
FROM fact_order o
JOIN dim_date od ON o.order_date_key = od.date_key
JOIN dim_date sd ON o.ship_date_key = sd.date_key;
```

**Common role-playing dimensions:**
- Date (order date, ship date, due date)
- Employee (salesperson, manager, approver)
- Location (origin, destination)
</details>

---

### Q29: What are the three layers of data warehouse architecture?
**Keywords:** Staging, Integration, Presentation, Landing Zone, Data Marts
<details>
<summary>Click to Reveal Answer</summary>

| Layer | Purpose | Characteristics |
|-------|---------|-----------------|
| **Staging** | Capture raw data from sources | Mirrors source structure, minimal transformation, temporary |
| **Integration** | Clean, transform, integrate | Business rules applied, conformed dimensions, historical data |
| **Presentation** | Serve users and BI tools | Star schemas, data marts, aggregates, views |

**Data flow:**
```
Sources -> Staging -> Integration -> Presentation -> BI Tools/Users
           (raw)      (cleaned)      (optimized)
```

**BigQuery example:**
```sql
-- Datasets for each layer
staging.raw_orders     -- As extracted from source
integration.fact_orders -- Cleaned, with surrogate keys
presentation.v_sales    -- User-facing views
```
</details>

---

### Q30: Compare the Kimball (bottom-up) and Inmon (top-down) data warehouse approaches.
**Keywords:** Data Marts First, Enterprise DW First, Methodology
<details>
<summary>Click to Reveal Answer</summary>

| Aspect | Kimball (Bottom-Up) | Inmon (Top-Down) |
|--------|---------------------|------------------|
| Build order | Data marts first | Enterprise DW first |
| Schema | Dimensional (star) | Normalized (3NF) |
| Time to value | Faster | Longer |
| Integration | Conformed dimensions | Central warehouse |
| Upfront design | Less | More |

**Kimball approach:**
```
Sources -> Data Mart 1 (Sales)    ---+
        -> Data Mart 2 (Inventory) --+--> Connected via conformed dims
        -> Data Mart 3 (HR)       ---+
```

**Inmon approach:**
```
Sources -> Enterprise Data Warehouse (normalized) -> Data Marts
```

**Modern practice:** Most organizations use Kimball-style dimensional modeling within cloud warehouses, with some Inmon concepts for enterprise integration.
</details>

---

### Q31: What is the Kimball four-step dimensional design process?
**Keywords:** Business Process, Grain, Dimensions, Facts
<details>
<summary>Click to Reveal Answer</summary>

**The Four Steps:**

1. **Choose the Business Process**
   - What are we measuring? (Sales, Orders, Inventory)

2. **Declare the Grain**
   - What does one row represent? (One sale, one day, one transaction)

3. **Identify the Dimensions**
   - Who, what, where, when, how? (Customer, Product, Store, Date)

4. **Identify the Facts**
   - What are we measuring? (Quantity, Revenue, Cost)

**Example:**
```
Business Process: Retail Sales
Grain: One row per line item on a sales receipt
Dimensions: Date, Customer, Product, Store, Promotion
Facts: Quantity sold, Unit price, Discount, Total amount
```

This systematic approach ensures consistent, well-designed dimensional models.
</details>

---

### Q32: How would you handle NULL values in dimension foreign keys?
**Keywords:** Unknown Member, Default Row, Data Quality, Missing Data
<details>
<summary>Click to Reveal Answer</summary>

Use an **"Unknown" dimension row** instead of NULL foreign keys.

**Implementation:**
```sql
-- Create unknown member in dimension
INSERT INTO dim_customer (customer_key, name, segment)
VALUES (-1, 'Unknown', 'Unknown');

-- Replace NULL FKs with unknown key
UPDATE fact_sales 
SET customer_key = -1 
WHERE customer_key IS NULL;
```

**Benefits:**
- Avoids NULL handling in queries
- All facts have valid dimension references
- Reports include all data (including unknown)
- Cleaner join behavior

**Variations:**
- `-1` for Unknown
- `0` for Not Applicable
- Separate unknown values for different missing data reasons
</details>

---

### Q33: Design a dimensional model for tracking employee performance reviews. Identify the grain, dimensions, and facts.
<details>
<summary>Click to Reveal Answer</summary>

**Business Process:** Employee Performance Reviews

**Grain:** One row per performance review per employee per review period

**Fact Table: `fact_performance_review`**
```sql
CREATE TABLE fact_performance_review (
    review_key INT64,           -- Surrogate key
    employee_key INT64,         -- FK to dim_employee
    reviewer_key INT64,         -- FK to dim_employee (role-playing)
    review_date_key INT64,      -- FK to dim_date
    review_period_key INT64,    -- FK to dim_review_period
    department_key INT64,       -- FK to dim_department
    -- Facts (measures)
    overall_rating NUMERIC(2,1),
    goal_completion_pct NUMERIC(5,2),
    competency_score NUMERIC(3,1),
    salary_increase_pct NUMERIC(5,2)
);
```

**Dimensions:**

| Dimension | Key Attributes |
|-----------|----------------|
| `dim_employee` | employee_key, name, hire_date, job_title, level |
| `dim_department` | department_key, name, manager, cost_center |
| `dim_date` | date_key, full_date, quarter, year |
| `dim_review_period` | period_key, period_name (Q1 2024), period_type (quarterly/annual) |

**Note:** `reviewer_key` is a role-playing dimension using the same `dim_employee` table.
</details>

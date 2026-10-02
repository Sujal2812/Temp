import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
OneHotEncoder,
StandardScaler
)

from sklearn.model_selection import (
train_test_split,
GridSearchCV,
cross_val_score
)

from sklearn.linear_model import (
LinearRegression
)

from sklearn.ensemble import (
RandomForestRegressor,
GradientBoostingRegressor
)

from sklearn.metrics import (
mean_squared_error
)


def train(df):

    target = "Optimal_Stock_Level"

    X=df.drop(
        target,
        axis=1
    )

    y=df[target]

    num=X.select_dtypes(
        include="number"
    ).columns

    cat=X.select_dtypes(
        exclude="number"
    ).columns


    prep=ColumnTransformer([

(
"num",
StandardScaler(),
num
),

(
"cat",
OneHotEncoder(
handle_unknown="ignore"
),
cat
)

])


    models={

"Linear":

LinearRegression(),

"RandomForest":

GridSearchCV(

RandomForestRegressor(),

{

"n_estimators":[50,100],

"max_depth":[5,10]

},

cv=3

),

"Gradient":

GradientBoostingRegressor()

}


    results={}

    xtr,xts,ytr,yts=(
        train_test_split(

X,
y,

test_size=.2,

random_state=42
)
)


    best=None

    best_rmse=99999

    best_pred=None


    for name,m in models.items():

        pipe=Pipeline([

("prep",prep),

("model",m)

])

        pipe.fit(
xtr,
ytr
)

        pred=pipe.predict(
xts
)

        rmse=np.sqrt(

mean_squared_error(
yts,
pred
)

)

        cv=(
cross_val_score(

pipe,

X,

y,

cv=5,

scoring="neg_root_mean_squared_error"

)
.mean()
*-1
)

        results[
name
]=[
rmse,
cv
]

        if rmse<best_rmse:

            best=pipe

            best_rmse=rmse

            best_pred=pred


    joblib.dump(

best,

"models/best_model.pkl"

)

    return (

results,

yts,

best_pred
)
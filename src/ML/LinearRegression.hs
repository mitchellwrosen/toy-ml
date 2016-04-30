{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ViewPatterns        #-}

module ML.LinearRegression
  ( TrainingData
  , Hypothesis
  , trainHypothesis
  , trainHypothesisDebug
  ) where

import Data.Matrix (Matrix)
import Data.Maybe  (fromJust)
import Data.Vector (Vector)
import Debug.Trace (traceShow)

import qualified Control.Foldl as Foldl
import qualified Data.Matrix   as Matrix
import qualified Data.Vector   as Vector

-- | Feature matrix, containing one example per row.
type TrainingData a = Matrix a

-- | A trained 'Hypothesis' outputs values given feature 'Vector's.
type Hypothesis a = Vector a -> a

type Params a = Vector a

-- | @trainHypothesis iters alpha xs ys@ trains a 'Hypothesis' using @iters@
-- iterations of gradient descent, with @α = alpha@.
--
-- @
-- > let training_xs = Matrix.fromLists [[2104,5,1,45],[1416,3,2,40],[1534,3,2,30],[852,2,1,36]]
-- > let training_ys = Vector.fromList [460,232,315,178]
-- > let hypothesis = trainHypothesis 1000 0.05 training_xs training_ys
-- > hypothesis (Vector.fromList [2000, 2, 2, 35])
-- 295.30300153265136
-- @
trainHypothesis
  :: forall a. (Show a, Ord a, Num a, Fractional a)
  => Int
  -> a
  -> TrainingData a
  -> Vector a
  -> Hypothesis a
trainHypothesis = trainHypothesis_ Nothing

-- | Like @trainHypothesis@, but takes an additional training sample to run
-- against each gradient descent step. Each iteration outputs
-- @expected - actual@, which should move towards 0.
--
-- @
-- > let training_xs = Matrix.fromLists [[2104,5,1,45],[1416,3,2,40],[1534,3,2,30],[852,2,1,36]]
-- > let training_ys = Vector.fromList [460,232,315,178]
-- > let x0 = Vector.fromList [2104,5,1,45]
-- > let y0 = 460
-- > let hypothesis = trainHypothesisDebug (x0,y0) 1000 0.05 training_xs training_ys
-- > hypothesis (Vector.fromList [2000, 2, 2, 35])
-- 620.4673421807009
-- 603.789203117362
-- 587.8599241747207
-- 572.6435731329733
-- ...
-- 181.3027228745516
-- 181.29971956831366
-- 181.2967232018583   // this is the last \'expected - actual\'
-- 295.30300153265136
-- @
trainHypothesisDebug
  :: forall a. (Show a, Ord a, Num a, Fractional a)
  => (Vector a, a)
  -> Int
  -> a
  -> TrainingData a
  -> Vector a
  -> Hypothesis a
trainHypothesisDebug = trainHypothesis_ . Just

trainHypothesis_
  :: forall a. (Show a, Ord a, Num a, Fractional a)
  => Maybe (Vector a, a)
  -> Int
  -> a
  -> TrainingData a
  -> Vector a
  -> Hypothesis a
trainHypothesis_ mdebug iters alpha xss ys =
  let
    mean_range_vector :: Vector (a, a)
    mean_range_vector = meanRangeVector xss

    -- Training data, scaled by subtracting mean and dividing by range.
    xss' :: TrainingData a
    xss' = Matrix.elementwise scaleElem mean_range_matrix xss
     where
      mean_range_matrix :: Matrix (a, a)
      mean_range_matrix = replicateRowVector (Matrix.nrows xss) mean_range_vector

    -- Scaled training data with a 1 prepended to every row
    xss'' :: TrainingData a
    xss'' = Matrix.colVector (Vector.replicate (Matrix.nrows xss) 1) Matrix.<|> xss'

    trainParams :: Int -> Params a -> Params a
    trainParams 0 ps = ps
    trainParams n ps =
      let
        ps'  = gradDescentStep xss'' ys alpha ps
        ps'' = trainParams (n-1) ps'
      in
        case mdebug of
          Nothing -> ps''
          Just (xs,y) -> traceShow (y - hypothesis ps' xs) ps''

    -- Train params by iterating gradient descent
    trained_params :: Params a
    trained_params = trainParams iters initial_params
     where
      initial_params :: Params a
      initial_params = Vector.replicate (Matrix.ncols xss'') 1

    hypothesis :: Params a -> Hypothesis a
    hypothesis ps v =
      Vector.cons 1 (fmap (uncurry scaleElem) (Vector.zip mean_range_vector v))
      `vdot`
      ps
  in
    hypothesis trained_params

-- | Perform one step of gradient descent, transforming the input params.
-- Assumes the training data is already scaled, and has 1s prepended to each
-- row.
gradDescentStep :: (Num a, Fractional a) => TrainingData a -> Vector a -> a -> Params a -> Params a
gradDescentStep xss ys a ts =
  Vector.zipWith (\t d -> t - a*d) ts (cost' ts xss ys)
 where
  cost' :: (Num a, Fractional a) => Params a -> TrainingData a -> Vector a -> Params a
  cost' ts xss ys =
    normalize (Matrix.getRow 1 (Matrix.transpose ds * xss))
   where
    -- Column vector
    ds = xss * Matrix.colVector ts - Matrix.colVector ys
    normalize = fmap (/ (fromIntegral (length ys)))

-- Matrix extras

vdot :: Num a => Vector a -> Vector a -> a
vdot xs ys = sum (Vector.zipWith (*) xs ys)

getCols :: Matrix a -> Vector (Vector a)
getCols m = Vector.fromList (map (\i -> Matrix.getCol i m) [1 .. Matrix.ncols m])

replicateRowVector :: Int -> Vector a -> Matrix a
replicateRowVector n0 (Matrix.rowVector -> m0) = go (n0-1) m0
 where
  go 0 acc = acc
  go n acc = go (n-1) (m0 Matrix.<-> acc)

meanRangeVector :: forall a. (Ord a, Num a, Fractional a) => TrainingData a -> Vector (a,a)
meanRangeVector xss = Foldl.fold ((,) <$> meanFold <*> rangeFold) <$> getCols xss
 where
  meanFold :: Foldl.Fold a a
  meanFold = (/)
    <$> Foldl.sum
    <*> Foldl.genericLength

  rangeFold :: Foldl.Fold a a
  rangeFold = (-)
    <$> (fromJust <$> Foldl.maximum)
    <*> (fromJust <$> Foldl.minimum)

scaleElem :: forall a. (Fractional a, Num a) => (a, a) -> a -> a
scaleElem (mean, range) x = (x-mean)/range

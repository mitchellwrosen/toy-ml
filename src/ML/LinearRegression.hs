{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module ML.LinearRegression
  ( trainHypothesis
  , gradientDescent
  ) where

import Control.Applicative (liftA2)
import Data.Maybe          (fromJust)
import GHC.TypeLits        (type (+), KnownNat)
import Linear.Matrix
import Linear.Metric
import Linear.V
import Linear.Vector

import qualified Control.Foldl as Foldl
import qualified Data.Vector   as Vector

-- Internal type aliases for documentation purposes.
type Mean a = a
type Range a = a

-- | @trainHypothesis iters alpha xs ys@ trains a hypothesis using @iters@
-- iterations of gradient descent, with @α = alpha@.
--
-- @
-- > let xs = V [V [2104,5,1,45], V [1416,3,2,40], V [1534,3,2,30], V [852,2,1,36]] :: V 4 (V 4 Double)
-- > let ys = V [460,232,315,178] :: V 4 Double
-- > let hypothesis = trainHypothesis 1000 0.05 xs ys
-- > hypothesis (V [2000,2,2,35])
-- 295.30300153265136
-- @
trainHypothesis
  :: forall n m a. (KnownNat n, KnownNat m, KnownNat (m+1), Fractional a, Num a, Ord a)
  => Int
  -> a
  -> V n (V m a)
  -> V n a
  -> V m a
  -> a
trainHypothesis iters alpha xs0 ys xs =
  let
    -- Means (u) and ranges (r) of each feature.
    urs :: V m (Mean a, Range a)
    urs = Foldl.fold (liftA2 (,) mean range) . toVector <$> transpose xs0
     where
      mean :: Foldl.Fold a (Mean a)
      mean = liftA2 (/) Foldl.sum Foldl.genericLength

      range :: Foldl.Fold a (Range a)
      range = liftA2 (-) (fromJust <$> Foldl.maximum) (fromJust <$> Foldl.minimum)

    scale :: (Mean a, Range a) -> a -> a
    scale (u,r) x = (x-u)/r

    -- Scale training data by subtracting mean and dividing by range, then
    -- prepend a 1 to every row.
    xs0' :: V n (V (m+1) a)
    xs0' = vcons 1 . liftI2 scale urs <$> xs0

    trained_params :: V (m+1) a
    trained_params = iterate (gradientDescent alpha xs0' ys) initial_params !! iters
     where
      -- A type-unsafe construction here; we assert that 1 + length xs = m + 1.
      initial_params :: V (m+1) a
      initial_params = V (Vector.replicate (1 + length xs) 1)
  in
    vcons 1 (liftI2 scale urs xs) `dot` trained_params

-- | Take one step using the gradient descent algorithm with learning rate @α@.
gradientDescent
  :: (KnownNat n, KnownNat m, Num a, Fractional a)
  => a
  -> V n (V m a)
  -> V n a
  -> V m a
  -> V m a
gradientDescent alpha xs ys ts = ts - alpha *^ (cost xs ys ts)

-- | Calculate the derivative of the cost function with respect to θ.
--
-- (δ/δ(θ_j))(J(θ)) = 1/n * Σ(h(θ,x)-y)(x_j)
--
cost
  :: (KnownNat n, KnownNat m, Num a, Fractional a)
  => V n (V m a)
  -> V n a
  -> V m a
  -> V m a
cost xs ys ts = ((xs !* ts - ys) *! xs) ^/ fromIntegral (length ys)

vcons :: a -> V n a -> V (n+1) a
vcons x (V xs) = V (Vector.cons x xs)

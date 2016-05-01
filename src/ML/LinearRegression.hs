{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module ML.LinearRegression
  ( trainHypothesis
  , scaleFeatures
  , gradientDescent
  , normal
  , vcons
  ) where

import ML.Internal.Shim    (pinv)

import Control.Applicative (liftA2)
import Data.Distributive   (Distributive)
import Data.Maybe          (fromJust)
import GHC.TypeLits        (type (+), KnownNat)
import Linear.Matrix
import Linear.Metric
import Linear.V
import Linear.Vector

import qualified Control.Foldl as Foldl
import qualified Data.Vector   as Vector

-- | @trainHypothesis iters alpha xs ys@ trains a hypothesis using @iters@
-- iterations of gradient descent, with @α = alpha@.
--
-- @
-- > let xs = V [V [2104,5,1,45], V [1416,3,2,40], V [1534,3,2,30], V [852,2,1,36]] :: V 4 (V 4 Double)
-- > let ys = V [460,232,315,178] :: V 4 Double
-- > let hypothesis = 'trainHypothesis' 1000 0.05 xs ys
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
    scaleRow :: V m a -> V m a
    scaleRow = scaleFeatures xs0

    -- Scale training data by subtracting mean and dividing by range, then
    -- prepend a 1 to every row.
    xs0' :: V n (V (m+1) a)
    xs0' = vcons 1 . scaleRow <$> xs0

    trained_params :: V (m+1) a
    trained_params = iterate (gradientDescent alpha xs0' ys) initial_params !! iters
     where
      -- A type-unsafe construction here; we assert that 1 + length xs = m + 1.
      initial_params :: V (m+1) a
      initial_params = V (Vector.replicate (1 + length xs) 1)
  in
    vcons 1 (scaleRow xs) `dot` trained_params

-- | Given a matrix of training data, return a row-scaling function that
-- subtracts the mean and divides by the range of each feature.
--
-- @
-- scaleFeatures :: (Fractional a, Ord a) => V n (V m a) -> V m a -> V m a
-- @
scaleFeatures
  :: forall a t f. (Fractional a, Ord a, Functor t, Foldable t, Distributive f, Additive f)
  => t (f a) -> f a -> f a
scaleFeatures xs = liftI2 scale urs
 where
  urs :: f (a, a)
  urs = Foldl.fold (liftA2 (,) mean range) <$> transpose xs
   where
    mean :: Foldl.Fold a a
    mean = liftA2 (/) Foldl.sum Foldl.genericLength

    range :: Foldl.Fold a a
    range = liftA2 (-) (fromJust <$> Foldl.maximum) (fromJust <$> Foldl.minimum)

  scale :: (a, a) -> a -> a
  scale (u,r) x = (x-u)/r

-- | Take one step using the gradient descent algorithm with learning rate @α@.
--
-- @
-- gradientDescent :: Fractional a => a -> V n (V m a) -> V n a -> V m a -> V m a
-- @
gradientDescent
  :: (Fractional a, Num (f a), Num (t a), Foldable f, Foldable t, Additive f, Additive t)
  => a
  -> t (f a)
  -> t a
  -> f a
  -> f a
gradientDescent alpha xs ys ts = ts - alpha *^ (cost xs ys ts)

-- | Calculate the derivative of the cost function with respect to θ.
--
-- (δ/δ(θ_j))(J(θ)) = 1/n * Σ(h(θ,x)-y)(x_j)
--
-- @
-- cost :: Fractional a => V n (V m a) -> V n a-> V m a -> V m a
-- @
cost
  :: (Fractional a, Num (t a), Foldable f, Foldable t, Additive f, Additive t)
  => t (f a)
  -> t a
  -> f a
  -> f a
cost xs ys ts = ((xs !* ts - ys) *! xs) ^/ fromIntegral (length ys)

-- | The normal equation.
--
-- @
-- θ = inv(X'X)X'y
-- @
--
-- @
-- > let xs = V [V [2104,5,1,45], V [1416,3,2,40], V [1534,3,2,30], V [852,2,1,36]] :: V 4 (V 4 Double)
-- > let scale = 'scaleFeatures' xs
-- > let xs' = 'vcons' 1 . scale <$> xs
-- > let ys = V [460,232,315,178] :: V 4 Double
-- > let ts = 'normal' xs' ys
-- > let hypothesis = \v -> 'dot' (vcons 1 (scale v)) ts
-- > hypothesis (V [2000,2,2,35])
-- 295.95655623793584
-- @
normal :: (Foldable t, Additive t, KnownNat n) => t (V n Double) -> t Double -> V n Double
normal xs ys = pinv (xs' !*! xs) !*! xs' !* ys
 where
  xs' = transpose xs

-- | Cons an element onto a 'V'.
vcons :: a -> V n a -> V (n+1) a
vcons x (V xs) = V (Vector.cons x xs)

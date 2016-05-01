{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module ML.LinearRegression
  ( trainHypothesis
  , gradientDescent
  , normal
  ) where

import ML.Internal.Shim    (pinv)

import Control.Applicative (liftA2)
import Data.Distributive   (Distributive)
import Data.Maybe          (fromJust)
import Data.Proxy          (Proxy(..))
import GHC.TypeLits        (type (+), KnownNat, Nat, natVal)
import Linear.Matrix
import Linear.Metric
import Linear.V
import Linear.Vector

import qualified Control.Foldl as Foldl
import qualified Data.Vector   as Vector

-- | @trainHypothesis method xs ys@ trains a hypothesis using training data
-- @xs@ and labels @ys@ with method @method@. The two methods provided by this
-- module are 'gradientDescent' and 'normal'.
--
-- @
-- -- Make some training data and labels.
-- > :set -XOverloadedLists
-- > let xs = 'V' ['V' [2104,5,1,45], 'V' [1416,3,2,40], 'V' [1534,3,2,30], 'V' [852,2,1,36]] :: 'V' 4 ('V' 4 Double)
-- > let ys = 'V' [460,232,315,178] :: 'V' 4 Double
--
-- -- Train a hypothesis using gradient descent, and test it out on a new example.
-- > let h1 = 'trainHypothesis' ('gradientDescent' 1000 0.05) xs ys
-- > h1 ('V' [2000,2,2,35])
-- 295.30300153265136
--
-- -- Train a hypothesis using the normal equation, and test it out on a new example.
-- > let h2 = 'trainHypothesis' 'normal' xs ys
-- > h2 ('V' [2000,2,2,35])
-- 295.95655623793584
-- @
trainHypothesis
  :: forall a (n :: Nat) (m :: Nat).
     (Fractional a, Num a, Ord a, KnownNat n, KnownNat m, KnownNat (m+1))
  => (forall i j. (KnownNat i, KnownNat j) => V i (V j a) -> V i a -> V j a)
  -> V n (V m a)
  -> V n a
  -> V m a
  -> a
trainHypothesis trainParams xs0 ys xs =
  let
    scaleRow :: V m a -> V m a
    scaleRow = scaleFeatures xs0

    xs0' :: V n (V (m+1) a)
    xs0' = vcons 1 . scaleRow <$> xs0

    params :: V (m+1) a
    params = trainParams xs0' ys
  in
    vcons 1 (scaleRow xs) `dot` params

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

-- | @gradientDescent iters alpha xs ys@ learns parameters @θ@ corresponding to
-- training data @xs@ and labels @ys@ by applying @iters@ iterations of the
-- gradient descent algorithm using learning rate @alpha@.
--
-- May be used as the first argument to 'trainHypothesis'.
gradientDescent
  :: forall a m n. (Fractional a, KnownNat n, KnownNat m)
  => Int
  -> a
  -> V n (V m a)
  -> V n a
  -> V m a
gradientDescent iters alpha xs ys =
  iterate (gradientDescentStep alpha xs ys) ts !! iters
 where
  ts :: V m a
  ts = V (Vector.replicate m 1)

  m :: Int
  m = fromIntegral (natVal (Proxy :: Proxy m))

-- | Take one step using the gradient descent algorithm with learning rate @α@.
--
-- @
-- gradientDescentStep :: Fractional a => a -> V n (V m a) -> V n a -> V m a -> V m a
-- @
gradientDescentStep
  :: (Fractional a, Num (f a), Num (t a), Foldable f, Foldable t, Additive f, Additive t)
  => a
  -> t (f a)
  -> t a
  -> f a
  -> f a
gradientDescentStep alpha xs ys ts = ts - alpha *^ (cost xs ys ts)

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

-- |
--
-- @
-- θ = inv(X'X)X'y
-- @
--
-- The normal equation. Calculates parameters @θ@ corresponding to
-- training data @xs@ and labels @ys@.
--
-- May be used as the first argument to 'trainHypothesis'.
normal :: (Foldable t, Additive t, KnownNat n) => t (V n Double) -> t Double -> V n Double
normal xs ys = pinv (xs' !*! xs) !*! xs' !* ys
 where
  xs' = transpose xs

-- | Cons an element onto a 'V'.
vcons :: a -> V n a -> V (n+1) a
vcons x (V xs) = V (Vector.cons x xs)

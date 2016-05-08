{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators       #-}

module ML.Regression where

import Control.Applicative (liftA2)
import Data.Distributive   (Distributive)
import Data.Maybe          (fromJust)
import Data.Proxy          (Proxy(..))
import GHC.TypeLits        (type (+), KnownNat, Nat, natVal)
import Linear.Matrix
import Linear.V
import Linear.Vector

import qualified Control.Foldl as Foldl
import qualified Data.Vector   as Vector

type Hypothesis m a =
  forall n. (KnownNat n, KnownNat m) => V n (V m a) -> V n a

-- | @trainHypothesis method hypothesis xs ys@ trains a hypothesis using training data
-- @xs@ and labels @ys@ with method @method@ and hypothesis @hypothesis@.
--
-- @
-- -- Make some training data and labels.
-- > import qualified ML.Regression.Linear as Linear
--
-- > :set -XOverloadedLists
-- > let xs = 'V' ['V' [2104,5,1,45], 'V' [1416,3,2,40], 'V' [1534,3,2,30], 'V' [852,2,1,36]] :: 'V' 4 ('V' 4 Double)
-- > let ys = 'V' [460,232,315,178] :: 'V' 4 Double
--
-- -- Train a hypothesis using gradient descent, and test it out on a new example.
-- > let h1 = 'trainHypothesis' ('gradientDescent' 1000 0.05) Linear.hypothesis xs ys
-- > h1 ('V' ['V' [2000,2,2,35]] :: 'V' 1 ('V' 4 Double))
-- V {toVector = [295.93124409794126]}
--
-- -- Train a hypothesis using the normal equation, and test it out on a new example.
-- -- We use 'const' here because the normal equation is not parameterized on the
-- -- hypothesis function, unlike gradient descent.
-- > let h2 = 'trainHypothesis' ('const' Linear.normal) Linear.hypothesis xs ys
-- > h2 ('V' ['V' [2000,2,2,35]] :: 'V' 1 ('V' 4 Double))
-- V {toVector = [295.95655623793584]}
-- @
trainHypothesis
  :: forall a (n :: Nat) (m :: Nat).
     (Fractional a, Num a, Ord a, KnownNat n, KnownNat m, KnownNat (m+1))
  => (forall i j. (KnownNat i, KnownNat j) => (V i (V j a) -> V j a -> V i a) -> V i (V j a) -> V i a -> V j a)
  -> (forall i j. (KnownNat i, KnownNat j) => V i (V j a) -> V j a -> V i a)
  -> V n (V m a)
  -> V n a
  -> Hypothesis m a
trainHypothesis trainParams h xs0 ys = \xs -> h (scale xs) params
 where
  scale :: forall r. KnownNat r => V r (V m a) -> V r (V (m+1) a)
  scale = fmap (vcons 1 . scaleFeatures xs0)

  params :: V (m+1) a
  params = trainParams h (scale xs0) ys

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

-- | @gradientDescent iters alpha h xs ys@ learns parameters @θ@ corresponding
-- to training data @xs@ and labels @ys@ using hypothesis @h(θ,x)@ by applying
-- @iters@ iterations of the gradient descent algorithm using learning rate
-- @alpha@.
gradientDescent
  :: forall a m n. (Fractional a, KnownNat n, KnownNat m)
  => Int
  -> a
  -> (V n (V m a) -> V m a -> V n a)
  -> V n (V m a)
  -> V n a
  -> V m a
gradientDescent iters alpha h xs ys =
  iterate (gradientDescentStep alpha h xs ys) ts !! iters
 where
  ts :: V m a
  ts = V (Vector.replicate m 1)

  m :: Int
  m = fromIntegral (natVal (Proxy :: Proxy m))

-- | Take one step using the gradient descent algorithm with hypothesis
-- @h(θ,x)@ and learning rate @α@.
--
-- @
-- gradientDescentStep
--   :: Fractional a
--   => (V n (V m a) -> V m a -> V n a)
--   -> a
--   -> V n (V m a)
--   -> V n a
--   -> V m a
--   -> V m a
-- @
gradientDescentStep
  :: (Fractional a, Num (f a), Num (t a), Foldable f, Foldable t, Additive f, Additive t)
  => a
  -> (t (f a) -> f a -> t a)
  -> t (f a)
  -> t a
  -> f a
  -> f a
gradientDescentStep alpha h xs ys ts = ts - alpha *^ cost' h xs ys ts

-- | Calculate the derivative of the cost function with respect to θ, using
-- hypothesis @h(θ,x)@.
--
-- @(δ/δ(θ_j))J(θ) = Σ(h(θ,x)-y)(x_j)@
--
-- @
-- cost
--   :: Fractional a
--   => (V n (V m a) -> V m a -> V n a)
--   -> V n (V m a)
--   -> V n a
--   -> V m a
--   -> V m a
-- @
cost'
  :: (Num a, Num (t a), Foldable t, Additive t, Additive f)
  => (t (f a) -> f a -> t a) -- ^ Hypothesis
  -> t (f a)                 -- ^ Training data
  -> t a                     -- ^ Labels
  -> f a                     -- ^ Parameters
  -> f a                     -- ^ Derivative of parameters
cost' h xs ys ts = (h xs ts - ys) *! xs

-- | Cons an element onto a 'V'.
vcons :: a -> V n a -> V (n+1) a
vcons x (V xs) = V (Vector.cons x xs)

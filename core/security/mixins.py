from core.security.crypto import encrypt_id

class SecureIDMixin:

    @property
    def secure_id(self):

        return encrypt_id(
            self.__class__.__name__,
            self.pk,
            self.organisation_id
        )